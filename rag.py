"""
RAG — Jour 1 : pipeline naïf
==================================

Étapes : chunk -> embed -> retrieve -> generate

Volontairement simple. Pas de framework (pas de LangChain/LlamaIndex) :
l'objectif du jour 1 est de VOIR ce qui se passe à chaque étape pour
comprendre où un RAG "naïf" casse sur des documents métier.

Pré-requis :
    ollama pull mistral
    ollama pull nomic-embed-text
    pip install -r requirements.txt
"""

import os
import re
import glob
import json
import sqlite3
import numpy as np
import ollama
from pypdf import PdfReader
from odf.opendocument import load as load_odt
from odf import text as odf_text, teletype

import claim_verification

CORPUS_DIR = "corpus"
GEN_MODEL = "mistral"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 500       # caractères par chunk (utilisé par le fallback naïf et le sub-split)
CHUNK_OVERLAP = 100    # chevauchement pour ne pas couper une idée en deux
MAX_CHUNK_SIZE = 1200  # taille max d'une section avant sub-split forcé

WHOLE_DOCUMENT_THRESHOLD = 1000
# Documents courts (ex: une fiche sinistre tenant en une lettre) sous ce seuil
# sont gardés comme UN SEUL chunk plutôt que découpés par paragraphe.
# Motivation (Jour 3) : un découpage par paragraphe peut séparer la date de
# survenance et la date de déclaration d'un même sinistre dans deux chunks
# différents — si top_k ne récupère qu'un des deux, le modèle ne peut
# physiquement pas comparer les deux dates, même avec des citations correctes.

TEMPERATURE = 0.0     # 0 = déterministe (toujours le token le plus probable).
                       # Mis à 0 après avoir observé des réponses contradictoires
                       # ("Oui" puis "Non" sur la même question + mêmes chunks) avec
                       # la température par défaut d'Ollama (~0.7-0.8, non documentée
                       # explicitement si on ne passe pas le paramètre).

TOP_K = 3              # nombre de chunks renvoyés au LLM

DB_PATH = "chunks.db"  # SQLite database for chunk storage


# ---------------------------------------------------------------------------
# METADATA LOADING
# ---------------------------------------------------------------------------

def load_metadata(metadata_file: str = "metadata.json") -> dict:
    """Load document metadata from JSON file."""
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    with open(metadata_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Build a dict: filename → metadata
    metadata_map = {doc["filename"]: doc for doc in data["documents"]}
    return metadata_map


def get_chunk_metadata(filename: str, metadata_map: dict) -> dict:
    """Get metadata for a chunk, or return defaults if not found."""
    if filename in metadata_map:
        meta = metadata_map[filename]
        return {
            "document_type": meta.get("document_type"),
            "contract_type": meta.get("contract_type"),
            "jurisdiction": meta.get("jurisdiction"),
        }
    # Fallback for files not in metadata.json
    return {
        "document_type": "unknown",
        "contract_type": None,
        "jurisdiction": None,
    }


# ---------------------------------------------------------------------------
# DATABASE INITIALIZATION
# ---------------------------------------------------------------------------

def init_database(db_path: str = DB_PATH) -> None:
    """Create SQLite database schema for chunks and embeddings."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            chunk_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            document_type TEXT,
            contract_type TEXT,
            jurisdiction TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def clear_database(db_path: str = DB_PATH) -> None:
    """Clear all chunks from database (called at start of each run)."""
    if os.path.exists(db_path):
        os.remove(db_path)
    init_database(db_path)


# ---------------------------------------------------------------------------
# 1. CHUNKING
# ---------------------------------------------------------------------------

SKIP_FILES = {
    "Code des assurances.pdf",  # version complète : trop volumineuse pour le jour 1,
                                  # on garde seulement les extraits ciblés (pages-3/4/5)
    "conditions_generales.txt",  # fichier vide, laissé par erreur
}


def clean_pdf_text(text: str) -> str:
    """
    Nettoyage basique du texte extrait d'un PDF :
    - normalise les espaces multiples
    - réduit les sauts de ligne excessifs (souvent dus à la mise en page PDF)
    - supprime les lignes très courtes isolées qui sont souvent des numéros
      de page ou des en-têtes/pieds de page répétés
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = text.split("\n")
    cleaned_lines = [line for line in lines if not re.fullmatch(r"\s*\d{1,4}\s*", line)]
    return "\n".join(cleaned_lines).strip()


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return clean_pdf_text("\n\n".join(pages_text))


def read_odt(path: str) -> str:
    doc = load_odt(path)
    paragraphs = doc.getElementsByType(odf_text.P)
    return "\n".join(teletype.extractText(p) for p in paragraphs).strip()


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents(corpus_dir: str) -> list[dict]:
    """Charge tous les documents du corpus (.md, .txt, .pdf, .odt)."""
    docs = []
    all_paths = sorted(
        glob.glob(os.path.join(corpus_dir, "*.md"))
        + glob.glob(os.path.join(corpus_dir, "*.txt"))
        + glob.glob(os.path.join(corpus_dir, "*.pdf"))
        + glob.glob(os.path.join(corpus_dir, "*.odt"))
    )
    if not all_paths:
        raise FileNotFoundError(
            f"Aucun document trouvé dans {corpus_dir}/ — vérifie le chemin."
        )

    for path in all_paths:
        filename = os.path.basename(path)
        if filename in SKIP_FILES:
            print(f"   (ignoré : {filename})")
            continue

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                text = read_pdf(path)
            elif ext == ".odt":
                text = read_odt(path)
            else:  # .md / .txt
                text = read_text_file(path)
        except Exception as e:
            print(f"   ⚠️  Erreur lecture {filename} : {e} — fichier ignoré")
            continue

        if not text.strip():
            print(f"   ⚠️  {filename} est vide après extraction — fichier ignoré")
            continue

        docs.append({"source": filename, "text": text})

    return docs


def chunk_text_naive(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Découpage naïf par nombre de caractères, avec chevauchement.
    Conservé comme référence de comparaison (jour 1) et comme filet de
    sécurité ultime si aucune limite de phrase n'est trouvée.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


SENTENCE_END_PATTERN = re.compile(r"[.!?]\s+")


def find_sentence_boundary(text: str, target_pos: int, search_window: int = 200) -> int:
    """
    Cherche la fin de phrase la plus proche de target_pos, en cherchant
    d'abord en arrière (pour ne jamais dépasser target_pos), puis en avant
    si rien n'est trouvé en arrière dans la fenêtre de recherche.

    Renvoie l'index juste après la ponctuation de fin de phrase trouvée,
    ou target_pos si aucune limite de phrase n'est trouvée dans la fenêtre
    (filet de sécurité : on coupe au mauvais endroit plutôt que de boucler
    indéfiniment ou produire un chunk démesuré).
    """
    window_start = max(0, target_pos - search_window)
    before = text[window_start:target_pos]
    matches = list(SENTENCE_END_PATTERN.finditer(before))
    if matches:
        last_match = matches[-1]
        return window_start + last_match.end()

    window_end = min(len(text), target_pos + search_window)
    after = text[target_pos:window_end]
    match = SENTENCE_END_PATTERN.search(after)
    if match:
        return target_pos + match.end()

    return target_pos


def chunk_text_by_sentence(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Découpage par taille cible, mais en ajustant chaque coupure pour tomber
    sur une fin de phrase plutôt qu'en plein milieu. Remplace le découpage
    naïf comme sous-diviseur des sections trop grandes (MAX_CHUNK_SIZE).
    """
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        target_end = start + chunk_size
        if target_end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        end = find_sentence_boundary(text, target_end)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        next_start = end - overlap
        # Garde-fou : toujours avancer pour éviter une boucle infinie
        # si la limite de phrase trouvée est trop proche du début.
        start = next_start if next_start > start else end

    return chunks


def chunk_by_markdown_headers(text: str) -> list[dict]:
    """
    Découpe un texte markdown en sections, une section = un header (#, ##, ###...)
    + tout le texte jusqu'au prochain header de niveau égal ou supérieur.

    Le header est gardé DANS le chunk : ça donne au LLM le contexte
    ("on est dans la section Expertise") même si le chunk est récupéré seul.

    Les sections "header seul" (un titre de section suivi immédiatement d'un
    sous-titre, sans texte propre — ex: "## 2. Déclaration du sinistre" suivi
    directement de "### 2.1 ...") sont fusionnées avec la section suivante
    plutôt que de devenir un chunk quasi vide.
    """
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(text))

    if not matches:
        # Pas de structure markdown détectée : retombe sur les paragraphes
        return chunk_by_paragraphs(text)

    MIN_SECTION_BODY_SIZE = 40  # en-dessous, on considère la section "header seul"

    raw_sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        header_title = match.group(2).strip()
        if section_text:
            raw_sections.append({"text": section_text, "header": header_title})

    # Texte avant le premier header (souvent le titre principal / préambule)
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            raw_sections.insert(0, {"text": preamble, "header": "préambule"})

    # Fusionne les sections "header seul" avec la section suivante
    sections = []
    pending_prefix = ""
    for section in raw_sections:
        body_only = section["text"].split("\n", 1)
        body_size = len(body_only[1].strip()) if len(body_only) > 1 else 0

        if body_size < MIN_SECTION_BODY_SIZE:
            # Pas de contenu propre : on garde le texte pour le préfixer à la suivante
            pending_prefix = f"{pending_prefix}\n\n{section['text']}".strip()
            continue

        if pending_prefix:
            section = {
                "text": f"{pending_prefix}\n\n{section['text']}".strip(),
                "header": section["header"],
            }
            pending_prefix = ""

        sections.append(section)

    # Si le document se termine sur un header sans contenu, on le rattache
    # à la dernière section plutôt que de le perdre.
    if pending_prefix:
        if sections:
            sections[-1]["text"] = f"{sections[-1]['text']}\n\n{pending_prefix}".strip()
        else:
            sections.append({"text": pending_prefix, "header": None})

    return sections


def chunk_by_paragraphs(text: str) -> list[dict]:
    """
    Découpe par paragraphes (séparés par une ou plusieurs lignes vides).
    Utilisé pour les PDF/ODT, qui n'ont pas de structure markdown
    après extraction (pypdf/odfpy perdent la mise en forme visuelle).

    Fusionne les paragraphes consécutifs très courts pour éviter des micro-chunks
    sans contexte suffisant (ex: une ligne de titre isolée).
    Le dernier fragment, s'il est trop court pour être un chunk autonome,
    est rattaché au chunk précédent plutôt que laissé seul (évite les
    micro-chunks de fin de document du type "9 PARIS." ou "le sinistre.").
    """
    raw_paragraphs = re.split(r"\n\s*\n", text)
    raw_paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    merged = []
    buffer = ""
    MIN_PARAGRAPH_SIZE = 150  # sous ce seuil, on fusionne avec le paragraphe suivant

    for p in raw_paragraphs:
        buffer = f"{buffer}\n\n{p}".strip() if buffer else p
        if len(buffer) >= MIN_PARAGRAPH_SIZE:
            merged.append(buffer)
            buffer = ""

    if buffer:
        if merged and len(buffer) < MIN_PARAGRAPH_SIZE:
            # Trop court pour être autonome : on le rattache au chunk précédent
            merged[-1] = f"{merged[-1]}\n\n{buffer}".strip()
        else:
            # Soit c'est le seul contenu du document, soit il n'y a rien à rattacher
            merged.append(buffer)

    return [{"text": p, "header": None} for p in merged]


def split_oversized_section(section: dict, max_size: int, overlap: int) -> list[dict]:
    """
    Si une section dépasse max_size, la sub-divise en respectant les limites
    de phrases (chunk_text_by_sentence) plutôt qu'en coupant à un nombre fixe
    de caractères. Le header est répété dans chaque sous-chunk pour garder
    le contexte.
    """
    if len(section["text"]) <= max_size:
        return [section]

    sub_texts = chunk_text_by_sentence(section["text"], max_size, overlap)
    return [{"text": t, "header": section["header"]} for t in sub_texts]


def chunk_document(text: str, is_markdown: bool) -> list[dict]:
    """
    Point d'entrée du chunking structure-aware.
    - Si le document entier est sous WHOLE_DOCUMENT_THRESHOLD : un seul chunk
      (évite de séparer des faits liés, ex: deux dates d'un même sinistre,
      dans des chunks différents — voir Jour 3 du README).
    - Markdown (.md) : découpe par headers
    - Autre (PDF/ODT extrait) : découpe par paragraphes
    Puis sub-divise toute section qui dépasse MAX_CHUNK_SIZE.
    """
    stripped = text.strip()
    if stripped and len(stripped) <= WHOLE_DOCUMENT_THRESHOLD:
        return [{"text": stripped, "header": None}]

    if is_markdown:
        sections = chunk_by_markdown_headers(text)
    else:
        sections = chunk_by_paragraphs(text)

    final_chunks = []
    for section in sections:
        final_chunks.extend(split_oversized_section(section, MAX_CHUNK_SIZE, CHUNK_OVERLAP))
    return final_chunks


def build_chunks(docs: list[dict]) -> list[dict]:
    """Transforme la liste de documents en liste de chunks avec leur source."""
    all_chunks = []
    for doc in docs:
        is_markdown = doc["source"].lower().endswith(".md")
        sections = chunk_document(doc["text"], is_markdown)
        for i, section in enumerate(sections):
            all_chunks.append(
                {
                    "source": doc["source"],
                    "chunk_id": i,
                    "text": section["text"],
                    "header": section["header"],
                }
            )
    return all_chunks


# ---------------------------------------------------------------------------
# 2. EMBEDDING
# ---------------------------------------------------------------------------

def embed_text(text: str) -> np.ndarray:
    """Appelle Ollama pour obtenir le vecteur d'embedding d'un texte."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return np.array(response["embedding"])


def embed_and_store_chunks(
    chunks: list[dict], metadata_map: dict, db_path: str = DB_PATH
) -> None:
    """Embed each chunk and store in SQLite database with metadata."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk["text"]).astype(np.float32)
        # Convert numpy array to bytes for storage
        embedding_bytes = embedding.tobytes()

        meta = get_chunk_metadata(chunk["source"], metadata_map)

        cursor.execute(
            """
            INSERT INTO chunks
            (source, chunk_id, text, embedding, document_type, contract_type, jurisdiction)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk["source"],
                chunk["chunk_id"],
                chunk["text"],
                embedding_bytes,
                meta["document_type"],
                meta["contract_type"],
                meta["jurisdiction"],
            ),
        )
        if (i + 1) % 50 == 0:
            print(f"   ...{i + 1} chunks ingérés")

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# 3. RETRIEVAL
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(
    query: str,
    top_k: int = TOP_K,
    contract_type_filter: str = None,
    document_type_filter: str = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """
    Hybrid retrieval: filter by metadata first, then rank by cosine similarity.
    
    Optional filters:
    - contract_type_filter: only return chunks from this contract type (e.g., 'dommages_eaux')
    - document_type_filter: only return chunks from this document type (e.g., 'conditions_generales')
    """
    query_vec = embed_text(query)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Build SQL query with optional metadata filters
    sql = "SELECT id, source, chunk_id, text, embedding, document_type, contract_type FROM chunks WHERE 1=1"
    params = []

    if contract_type_filter:
        sql += " AND contract_type = ?"
        params.append(contract_type_filter)
    if document_type_filter:
        sql += " AND document_type = ?"
        params.append(document_type_filter)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return []

    # Score all retrieved rows by cosine similarity
    scored = []
    for row in rows:
        chunk_id, source, chunk_num, text, embedding_bytes, doc_type, contract_type = (
            row
        )
        # Reconstruct numpy array from bytes
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

        score = cosine_similarity(query_vec, embedding)
        scored.append(
            {
                "source": source,
                "chunk_id": chunk_num,
                "text": text,
                "score": score,
                "document_type": doc_type,
                "contract_type": contract_type,
            }
        )

    # Return top-k
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# 4. GENERATION
# ---------------------------------------------------------------------------

def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Construit le prompt en assignant un identifiant court à chaque chunk
    (ex: [A], [B], [C]) et en demandant explicitement au modèle de citer
    cet identifiant après CHAQUE fait avancé dans sa réponse.

    Objectif : rendre visible, fait par fait, quelle source justifie quelle
    affirmation — pour détecter les cas où un fait correct est associé au
    mauvais document (contamination inter-documents, cf. README Jour 3).
    """
    context_blocks = []
    chunk_ids = []
    for i, c in enumerate(retrieved_chunks):
        chunk_id = chr(65 + i)  # A, B, C, ...
        chunk_ids.append(chunk_id)
        context_blocks.append(
            f"[{chunk_id}] (source : {c['source']} — extrait #{c['chunk_id']})\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    valid_ids = ", ".join(f"[{cid}]" for cid in chunk_ids)

    return f"""Tu es un assistant pour des gestionnaires sinistres.
Réponds à la question UNIQUEMENT à partir du contexte fourni ci-dessous.

RÈGLES DE CITATION (obligatoires) :
1. Chaque source du contexte a un identifiant entre crochets : {valid_ids}
2. Après CHAQUE fait que tu avances (une date, un nom, un montant, une règle),
   ajoute immédiatement l'identifiant de la source qui le justifie, ex: "le 14/01/2025 [A]".
3. Si un fait nécessaire à la réponse n'apparaît dans AUCUNE source, dis-le
   explicitement plutôt que de l'inventer ou de le déduire.
4. N'attribue jamais un fait à une source qui ne le contient pas réellement —
   vérifie mentalement que le fait cité figure bien dans la source indiquée
   avant de l'écrire.
5. Si la question porte sur une personne ou un sinistre nommé, n'utilise que
   les faits provenant de la source qui mentionne CETTE personne précisément —
   ne mélange jamais les faits de deux personnes différentes même si leurs
   dossiers se ressemblent.

Contexte :
{context}

Question : {query}

Réponse (avec citation [{chunk_ids[0] if chunk_ids else 'A'}] après chaque fait) :"""


def generate(query: str, retrieved_chunks: list[dict]) -> str:
    prompt = build_prompt(query, retrieved_chunks)
    response = ollama.generate(
        model=GEN_MODEL,
        prompt=prompt,
        options={"temperature": TEMPERATURE},
    )
    return response["response"]


# ---------------------------------------------------------------------------
# PIPELINE COMPLET
# ---------------------------------------------------------------------------

def detect_entity_in_query(query: str, facts_path: str = "claim_facts.json") -> str | None:
    """
    Cherche si le nom d'une personne connue (présente dans claim_facts.json)
    apparaît dans la question posée. Renvoie le premier nom de famille trouvé,
    ou None. Détection volontairement simple (sous-chaîne du nom de famille) :
    le but n'est pas de remplacer la compréhension du LLM, mais de déclencher
    une vérification déterministe EN PLUS de sa réponse, jamais à sa place —
    voir README, "Why Not Just Detect Deadline Questions", pour la discussion
    sur les limites d'une détection par mots-clés.
    """
    try:
        facts = claim_verification.load_claim_facts(facts_path)
    except FileNotFoundError:
        return None

    query_normalized = (
        query.lower().replace("è", "e").replace("é", "e").replace("ê", "e")
    )
    seen_names = set()
    for claim in facts["claims"]:
        full_name = claim["entity_name"]
        last_name = full_name.split()[-1]
        last_name_normalized = (
            last_name.lower().replace("è", "e").replace("é", "e").replace("ê", "e")
        )
        if last_name_normalized in query_normalized and full_name not in seen_names:
            seen_names.add(full_name)
            return last_name
    return None


def main():
    print(f"📋 Chargement des métadonnées depuis 'metadata.json'...")
    metadata_map = load_metadata("metadata.json")
    print(f"   -> {len(metadata_map)} document(s) référencé(s) dans les métadonnées\n")

    print(f"📂 Chargement du corpus depuis '{CORPUS_DIR}/'...")
    docs = load_documents(CORPUS_DIR)
    print(f"   -> {len(docs)} document(s) chargé(s) : {[d['source'] for d in docs]}\n")

    print("✂️  Découpage en chunks...")
    chunks = build_chunks(docs)
    print(f"   -> {len(chunks)} chunks créés (taille {CHUNK_SIZE} caractères, "
          f"overlap {CHUNK_OVERLAP})\n")

    print("🗄️  Réinitialisation de la base de données...")
    clear_database(DB_PATH)

    print(f"🧮 Calcul des embeddings ({EMBED_MODEL}) et stockage en base...")
    embed_and_store_chunks(chunks, metadata_map, DB_PATH)
    print("   -> Embeddings calculés et stockés.\n")

    print("=" * 70)
    print("Pipeline prêt. Tape une question (ou 'exit' pour quitter).")
    print("Filtres optionnels : tape 'contrat:dommages_eaux ta question'")
    print("                 ou : tape 'type:procedure_interne ta question'")
    print("=" * 70)

    while True:
        raw_input_str = input("\n❓ Question : ").strip()
        if raw_input_str.lower() in {"exit", "quit", ""}:
            break

        # Parse optional inline filters: "contrat:xxx" or "type:xxx"
        contract_filter = None
        doc_type_filter = None
        query = raw_input_str

        if query.startswith("contrat:"):
            parts = query.split(" ", 1)
            contract_filter = parts[0].replace("contrat:", "")
            query = parts[1] if len(parts) > 1 else ""
        elif query.startswith("type:"):
            parts = query.split(" ", 1)
            doc_type_filter = parts[0].replace("type:", "")
            query = parts[1] if len(parts) > 1 else ""

        retrieved = retrieve(
            query,
            top_k=TOP_K,
            contract_type_filter=contract_filter,
            document_type_filter=doc_type_filter,
        )

        print("\n🔍 Chunks récupérés :")
        if not retrieved:
            print("   (aucun résultat — vérifie ton filtre)")
        for i, c in enumerate(retrieved):
            citation_id = chr(65 + i)
            preview = c["text"][:120].replace("\n", " ")
            tags = f"[{c['document_type']}/{c['contract_type']}]"
            print(f"   [{citation_id}] {c['source']} #{c['chunk_id']} {tags} (score={c['score']:.3f}) : {preview}...")

        print("\n🤖 Génération de la réponse...\n")
        answer = generate(query, retrieved)
        print(f"💬 {answer}")

        # Vérification déterministe (indépendante du LLM) si la question
        # mentionne une personne connue dans claim_facts.json. Affichée
        # SÉPARÉMENT de la réponse du LLM, pour permettre la comparaison
        # directe plutôt que de fusionner les deux sources de vérité.
        detected_name = detect_entity_in_query(raw_input_str)
        if detected_name:
            print(f"\n🧮 Vérification déterministe (claim_facts.json) pour '{detected_name}' :")
            results = claim_verification.verify_entity_deadline(detected_name)
            for r in results:
                if r["compliant"] is None:
                    print(f"   ⚠️  {r['reason']}")
                else:
                    verdict = "CONFORME" if r["compliant"] else "NON CONFORME"
                    print(f"   [{r.get('matched_filename', '?')}] {verdict} — {r['reason']}")


if __name__ == "__main__":
    main()
