# Sukoshii — RAG

Small RAG pipeline for querying insurance documents.  
Handles `.txt`, `.md`, `.pdf`, and `.odt` formats.

Built to understand **why naive RAG fails on regulated insurance documents**, not just that it does — and to fix those failures one at a time, verifying each fix actually does what it claims.

**Status:** Jour 1 (naive baseline) + Jour 2 (metadata/hybrid retrieval + structure-aware chunking + deterministic generation) + Jour 3 (citation tracking, whole-document chunking fix, and a deterministic verification layer that caught the LLM giving a confidently wrong answer) complete. See "Conclusion" below for the throughline across all three days.

---

## Quick Start

```bash
pip install -r requirements.txt
python3 rag.py
```

Then ask questions about your corpus interactively. The pipeline prints retrieved chunks and their similarity scores before generating answers, so you can see exactly what context the model is working from.

---

## Architecture

**Pipeline:** Document loading → Chunking → Embedding + metadata tagging → Hybrid retrieval (metadata filter + cosine similarity) → Generation (Mistral)

**Design choices:**
- **Chunking:** Structure-aware — markdown headers for `.md` sources, paragraph-based for PDF/ODT (which lose visual structure on extraction), with sentence-boundary-aware sub-splitting for oversized sections, and a whole-document bypass for short texts (see "Jour 2, Part 2" and "Jour 3, Part 3" below for why this took several iterations)
- **Metadata:** `metadata.json` maps each source file to `document_type`, `contract_type`, `jurisdiction` — kept as a separate file from the code on purpose, so the corpus can be re-tagged without touching pipeline logic
- **Storage:** SQLite (`chunks.db`) holds chunk text, embeddings (as bytes), and metadata columns — enables SQL-filtered retrieval, not just brute-force vector search over everything
- **Embeddings:** `nomic-embed-text` via Ollama (768-dim, locally hosted)
- **Retrieval:** Hybrid — optional metadata filter (`contract_type`, `document_type`) applied in SQL first, then top-k cosine similarity within the filtered set
- **Generation:** Mistral 7B via Ollama, `temperature=0` (deterministic) — set explicitly after finding contradictory answers on identical input at Ollama's default temperature (see Jour 2, Part 3)
- **Citation:** Each retrieved chunk gets a short letter ID (`[A]`, `[B]`, ...); the prompt requires the model to tag every factual claim with the ID of the source that supports it — added in Jour 3 to catch facts being attributed to the wrong document
- **Deterministic verification:** `claim_facts.json` holds hand-extracted per-claim facts (entity name, loss date, declaration date, applicable deadline rule); `claim_verification.py` computes deadline compliance with `numpy.busday_count`, entirely independent of the LLM. Triggered automatically when a question mentions a known name, and displayed alongside (not instead of) the LLM's answer — see Jour 3, Part 4 for why this exists

No framework abstractions (no LangChain, no LlamaIndex). The goal is **visibility**: every step prints what it's doing, so you can debug when things go wrong — which they did, repeatedly, and that's the point (see below).

---

## Key Findings (Jour 1)

### 1. Lexical Similarity ≠ Contextual Relevance

**Observed in:** "Quel est le délai de déclaration d'un sinistre dégât des eaux ?"

The model retrieved rules about declaring a *theft* (2-day deadline) alongside rules about declaring *water damage* (5-day deadline). Both matched on the word "délai" (deadline) and scored similarly, so retrieval grabbed them together. The model then confidently answered with **both timelines mixed**, even though one is completely wrong for the question asked.

**Root cause:** Cosine similarity on raw text embeddings has no concept of document type, contract scope, or semantic domain. A rule about auto insurance theft shares embedding space with a rule about home insurance water damage if they use similar vocabulary.

**Implication for insurance:** In a regulated domain where a 2-day vs. 5-day error changes compliance, this is a serious failure. Lexical overlap is not enough.

---

### 2. Retrieval Confidence ≠ Answer Confidence

**Observed in:** "Les sinistres mentionnés... sont-ils déclarés dans les délais impartis ?"

The model retrieved chunks about declaration *rules* (what customers must do), not chunks about actual *declarations* (specific sinistre dates). It then answered "Oui, les sinistres sont déclarés dans les délais" — confidently asserting a fact it had no data to check. The retrieved context proved the rule exists, not that anyone followed it.

**Root cause:** The question asked for a compliance *judgment* (is this specific claim compliant?), but retrieval only found statements of the *rule*. The model couldn't tell the difference and answered anyway.

**Implication for insurance:** Confusing "I found the regulation" with "I verified compliance" is exactly the kind of hallucination that breaks in audit or litigation scenarios.

---

### 3. Answering an Easier Sub-Question Unintentionally

**Observed in:** "Monsieur Dupont a-t-il déclaré son dégâts des eaux dans les temps ?"

Model answered "Oui, il a déclaré le 19/09/2021" — a factually correct extraction of the declaration date from the right document. **But** the actual question required a *comparison*: is `date_declared` ≤ `date_of_loss + 5 jours ouvrés`? 

The model retrieved the declaration date but not the loss date, so it answered "when was it declared" instead of "was it timely." The user would have to do the math themselves.

**Root cause:** Top-k retrieval has no way to know that answering this question requires *two pieces of related information*. It grabbed the most similar chunks, not the most complete answer.

**Implication for insurance:** Questions involving deadlines, comparisons, or multi-step logic silently degrade to partial answers, and the model sounds confident about incomplete data.

---

### 4. When It Works

**Observed in:** "Quel est le seuil de dispense d'expertise et qu'est-ce qui se passe au-delà ?"

Retrieved the right chunk cleanly, boundaries didn't cut through the logic, and the answer was accurate and complete. This is the **happy path**: small, self-contained, well-placed information.

---

## Jour 2, Part 1: Metadata + Hybrid Retrieval

**Fix:** Each document is tagged in `metadata.json` with `document_type`, `contract_type`, and `jurisdiction`. Retrieval can now filter by these tags *before* ranking by cosine similarity, instead of searching the entire corpus blindly.

**Result — same question, with and without the filter:**

| | Query | Top retrieved source | Answer |
|---|---|---|---|
| Without filter | "Quel est le délai de déclaration ?" | `cg-auto-614.pdf` (auto CG) | Mixes 2-day theft deadline with 30-day catastrophe naturelle deadline — wrong for the actual question |
| With `contract_type=dommages_eaux` filter | Same question | `note_procedure_degat_eaux.md.pdf` + `Sinistre degats des eaux.odt` | Clean, correct: 5 jours ouvrés, sourced to L113-2 |

This directly fixes **Finding #1** from Jour 1 (lexical similarity ≠ contextual relevance). The embeddings still can't distinguish "auto" from "habitation" on their own — but they no longer have to, because the SQL filter removes irrelevant contract types before similarity scoring even runs.

**Honest caveat:** this only works because *I* manually tagged each file in `metadata.json`. The pipeline still has no way to auto-detect that a question is about `dommages_eaux` and apply the filter itself — the filter has to be typed explicitly (`contrat:dommages_eaux <question>`). Automatic query classification (inferring the right filter from the question) is a real next step, not yet built.

---

## Jour 2, Part 2: Structure-Aware Chunking — Three Bugs, Not One

This is the part worth reading carefully, because the path to a working fix involved finding and fixing three separate, layered bugs — not one clean implementation. Documenting the process, not just the result, because the process is the actual lesson.

**Goal:** replace the day-1 naive character-slicer (cuts every 500 chars, blind to sentence/paragraph structure) with something that respects document structure, so numeric thresholds and their explanations don't get severed by an arbitrary boundary.

**Approach:** markdown header-based chunking for `.md` sources; paragraph-based chunking for PDF/ODT (since `pypdf`/`odfpy` extraction loses visual formatting — bold headers become plain text indistinguishable from body text); a size cap (`MAX_CHUNK_SIZE`) with sub-splitting for any section that ends up too large.

**Bug #1 — stale file, not a logic bug.** First test showed *zero* improvement (same 2025 chunks, same 500-char max) after the rewrite. Root cause: the new code was sitting in the wrong location relative to what Python was actually importing — a pure deployment mistake, not a flaw in the new chunking logic. Lesson: always verify the file being executed matches the file you think you edited (`python3 -c "import rag; print(rag.__file__)"` plus a `grep` for a function name unique to the new version) before debugging the logic itself.

**Bug #2 — orphan fragments from the oversized-section fallback.** Once the real code was running, chunk count dropped sensibly (2025 → 857) and average size improved — but ~12 chunks were tiny, meaningless fragments like `'le sinistre.'` or `'9 PARIS.'`. Root cause: when a paragraph-based section exceeded `MAX_CHUNK_SIZE`, the fallback sub-splitter (`chunk_text_naive`) cut it at a fixed character count — the exact same blind character-slicing the rewrite was supposed to eliminate, just relocated to a different code path. A fix at the document level doesn't help if the same naive logic still runs underneath it as a "safety net." **Fix:** replaced the naive sub-splitter with a sentence-boundary-aware version (`chunk_text_by_sentence`) that searches backward from the target cut point for the nearest sentence-ending punctuation, instead of cutting wherever the character count happens to land.

**Bug #3 — header-only sections.** Fixing #2 surfaced a second, unrelated issue on stress-testing: when a markdown section header (e.g. `## 2. Déclaration du sinistre`) had no content of its own before its first subsection began, it became its own near-empty chunk. Not a splitting bug — a structural one, specific to how markdown headers nest. **Fix:** sections with negligible body text (`< 40 chars`) are now merged forward into the next section instead of standing alone.

**Verification, not just claims:**
- Stress-tested the sentence-boundary splitter against empty text, no-punctuation text (3000 chars of `'a'`), and dense-punctuation text — no infinite loops, no crashes
- Confirmed zero word loss between original document text and reconstructed chunk text (set difference check)
- Re-ran the tiny-chunk check on the full real corpus after all three fixes: **0 chunks under 30 characters**, minimum chunk size 103 chars, max 1391 chars (target was 1200 — slightly over is expected, since sentence-boundary search can overshoot slightly to avoid a mid-sentence cut)

**Why this is worth keeping in the README as-is, bugs included:** a fix at one architectural layer (paragraph/header-level chunking) can silently fail to propagate to a fallback layer (the oversized-section sub-splitter) if that fallback wasn't rewritten with the same care. This is a general lesson about iterating on a pipeline, not just a chunking-specific one — and it's exactly the kind of thing that's invisible until you actually go looking for tiny/malformed outputs instead of just trusting that "the chunk count went down, so it must be better."

---

## Jour 2, Part 3: Re-testing Jour 1's Findings — A New Bug Hiding Behind an Old One

With metadata filtering and structure-aware chunking both in place, I went back and re-ran Jour 1's Finding #3 question (Monsieur Dupont) to see whether the retrieval-layer fixes had any effect on it. They hadn't — but re-testing surfaced a different, more fundamental bug that had nothing to do with retrieval at all.

**Re-running the same question, same retrieved chunks, two runs:**

| Run | Retrieved chunks | Answer |
|---|---|---|
| 1 | Identical — `Sinistre degats des eaux.odt` top result (score 0.771), correct document | "**Non**, Monsieur Dupont n'a pas déclaré..." |
| 2 | Identical — same chunks, same scores | "**Oui**, Monsieur Dupont a déclaré un dégât des eaux le 19 septembre 2021..." |

Same input, opposite conclusions. This ruled out a retrieval or chunking explanation immediately — if the model were confusing the two different "DUPONT Roger" entries that genuinely exist in the corpus (one in the water-damage file, one in an unrelated cambriolage file at a different address), it would fail *consistently* in the same direction. A coin-flip between "Oui" and "Non" on identical context instead points to **generation-level sampling randomness**.

**Root cause confirmed:** `ollama.generate()` was being called with no explicit options, meaning Mistral ran at Ollama's default temperature (non-zero, samples randomly). **Fix:** added an explicit `TEMPERATURE = 0.0` constant, passed via `options={"temperature": TEMPERATURE}`. Re-ran the same question multiple times after the fix — consistent answer every time.

**But fixing the coin-flip surfaced what it was actually masking.** With temperature locked at 0, the model consistently answers "Oui, déclaré le 19/09/2021" — which I confirmed against the source document is *correct*, but for an unintended reason: both the date of loss ("survenu le 19/09/2021") and the date of declaration ("Fait à Paris, le 19/09/2021") happen to be the **same day** in this particular fiche sinistre. The model still never explicitly retrieves the loss date, compares it to the declaration date, or checks the 5-day rule from the procedure note — it just states the most prominent date in the chunk it has. It got the right answer here purely because the corpus made the comparison trivial (same-day filing can't violate a 5-day window), not because the pipeline gained any comparison capability.

**Why this distinction matters:** "No date/timeline reasoning" was already a listed limitation after Jour 1, but it was a theoretical concern. Now there's concrete evidence of exactly how it fails silently: **a correct answer for the wrong reason**, indistinguishable from a correct answer for the right reason unless you already know the underlying dates yourself. If the loss date and declaration date had differed by, say, 6 days, this same shallow behavior would likely have produced an equally confident — and wrong — "Oui." The corpus didn't happen to test that case, so the gap stayed hidden until now.

**Two findings from one re-test, properly separated:**
1. **Non-determinism in generation** — real bug, fixed (temperature=0)
2. **No actual date-comparison logic** — still open, now demonstrated rather than assumed, and now precisely characterized: the failure mode is "answers confidently using only the most salient date in context," not "fails to find any date at all"

---

## Jour 3, Part 1: Built a Real Test Case — Found Something Worse Than Expected

Jour 1 Finding #3 (Monsieur Dupont) never actually tested date-comparison capability, because the loss date and declaration date happened to be identical in that document — "Oui" was correct by coincidence, not by computation. To find out whether the gap was luck or capability, I built two new synthetic fiches sinistre with **genuinely different** loss and declaration dates:

- **Martin** — loss `09/01/2025`, declared `14/01/2025` (5 days later) → should read as **compliant**
- **Lefèvre** — loss `03/02/2025`, declared `15/02/2025` (12 days later) → should read as **non-compliant**

**Both verdicts came back correct:**

| Question | Answer | Verdict correct? |
|---|---|---|
| Martin dans les temps ? | "Oui... car le sinistre a eu lieu le 09/01/2025" | ✅ Yes |
| Lefèvre dans les temps ? | "Non... car le sinistre a eu lieu le 03/02/2025 et la déclaration est datée du 19/09/2021" | ✅ Yes |

At first glance this looks like a pass. **It isn't.** Look at the justification for Lefèvre: *"la déclaration est datée du 19/09/2021."* Lefèvre's actual declaration date is `15/02/2025` — `19/09/2021` is **Monsieur Dupont's** declaration date, from a completely unrelated document that happened to also be retrieved in the same context window (three sinistre files all score similarly on a generic "dégât des eaux... dans les temps" query, so all three end up in top-k together).

**What actually happened:** the model reached the right verdict ("Non") while citing a fabricated link between two unrelated people's documents. `19/09/2021` vs `03/02/2025` isn't even a coherent date comparison — different years, no real elapsed time represented — yet it was presented as the basis for a confident, specific-sounding conclusion.

**Why this is a more serious finding than "no date arithmetic," which is what I expected to find:** a model that hedges or admits it can't compute a date gap is at least honestly limited. A model that **cross-attributes a fact from one client's file to another client's claim**, while sounding fully confident and citing a specific date, is actively dangerous in a compliance context — that's the kind of error that could end up in a written justification to a client or regulator without anyone noticing, because the sentence reads as if it makes sense.

**Martin's justification has the same shape, just less visible:** "Oui... car le sinistre a eu lieu le 09/01/2025" never mentions the declaration date or the gap at all — it's an assertion with a citation that doesn't actually support the conclusion drawn. It happened to be paired with the right verdict here, but the justification given doesn't establish that verdict; it just states one fact and one conclusion next to each other.

**This sharpens the Jour 2 finding considerably.** It's no longer "the model picks the most salient date instead of comparing two." It's "the model will pull a confident-sounding fact from *any* chunk in context, including ones describing a different person's claim entirely, if it lexically fits the sentence being constructed." That's a retrieval-adjacent failure (multiple similar documents in the same context) compounding a generation failure (no verification that a cited fact actually belongs to the entity being discussed) — and it's exactly the kind of error that citation tracking (already on the roadmap) would catch immediately, since forcing the model to link each fact to its source chunk would expose the mismatch instantly.

---

## Jour 3, Part 2: Prompt-Level Citation — Fixes Contamination, Doesn't Fix Verification

**Fix:** Rewrote `build_prompt()` to assign each retrieved chunk a short letter ID (`[A]`, `[B]`, `[C]`) and require the model to cite the source ID after every factual claim, with an explicit rule never to mix facts between different named entities even if their files look similar.

**Result on the same Lefèvre question that exposed the contamination bug:**

> "Non, Monsieur Lefevre n'a pas déclaré son dégât des eaux dans les temps, car le sinistre qu'il a subi est mentionné dans la source [B], et il y est indiqué que le dégât a eu lieu le 03/02/2025. Or, dans cette même source, il n'y a pas de déclaration de sinistre antérieure à cette date."

**The contamination is gone** — `[B]` is genuinely Lefèvre's own document, and the model correctly stopped borrowing Dupont's date. That part of the fix worked as intended.

**But this answer is also wrong, for a new and informative reason.** It claims the declaration date isn't in `[B]` — when in fact it was, just in a *different chunk* of the same document (`#0`, containing "Fait à Bordeaux, le 15/02/2025") than the one retrieved (`#1`, containing only the loss date). Diagnosis: my original paragraph-based chunker split this short letter into 4 separate chunks, scattering the two dates needed for the comparison across different chunks — so even with perfect citation discipline, the model could only ever see one of the two facts it needed, depending on which chunk `top_k` happened to retrieve.

**This is Jour 1 Finding #3, recurring in a more precise form.** Back then, the Dupont letter happened to have loss date = declaration date, so the missing comparison never got tested. Now, with two letters that have genuinely different dates, the *same underlying gap* — chunking can separate two facts that belong together — produces a visible, concrete failure instead of a lucky pass.

---

## Jour 3, Part 3: Whole-Document Chunking Fix, and a Sturdier Failure

**Fix:** Short documents (≤ `WHOLE_DOCUMENT_THRESHOLD = 1000` characters — calibrated to comfortably cover a one-page fiche sinistre letter, ~950 chars in testing) are now kept as a single chunk instead of being split by paragraph. This guarantees that any two related facts within a short document (loss date and declaration date, in this case) are always retrieved together.

**Verified before merging:** confirmed the fix doesn't regress larger structured documents — `note_procedure_degat_eaux.md` (5625 chars, well above the threshold) still produces its full 13-chunk structural breakdown, headers and all.

**Result after the fix — same two questions, re-run:**

| Question | Answer |
|---|---|
| Martin dans les temps ? | "Oui... la déclaration a été faite le 14/01/2025 [A]." |
| Lefèvre dans les temps ? | "Oui, Monsieur Lefèvre a déclaré son dégât des eaux dans les temps. ([A])" |

Martin's verdict is correct, Lefèvre's is now **wrong** — flipped from the correct "Non" to an incorrect "Oui." And this is the most important result of the whole day: **the citation is now valid** (`[A]` genuinely is Lefèvre's document, and it genuinely contains both dates after the chunking fix) **but the conclusion drawn from it is still unsupported.** No date gap is computed or shown in either answer; Martin's response doesn't even mention the loss date, and Lefèvre's offers no reasoning at all beyond a bare citation tag.

**Why this is a sturdier, more dangerous failure than the contamination bug it replaced:** a wrong citation (the original bug) is checkable by anyone willing to open the cited document and compare. A *valid* citation attached to an *unjustified* conclusion looks legitimate on inspection — the source really does say what's claimed, it just doesn't establish what the model concluded from it. This is much harder to catch by spot-checking, which is exactly the property that makes it more dangerous in a real compliance workflow, not less.

**Conclusion: citation tracking solved attribution, not verification.** Knowing *which* document a fact came from doesn't guarantee the model actually performed the comparison that document's facts were supposed to support. Those are two different problems, and Jour 1 through Jour 3 collapsed them into one assumption ("if retrieval finds the right facts, the model will reason over them correctly") that turned out to be false at every layer tested.

---

## Why Not Just "Detect Deadline Questions and Trigger a Date Check"?

The natural next idea is to detect when a question is asking about a deadline and run a deterministic date computation as a safety net. The date math itself is trivial and completely reliable — `datetime` subtraction has no hallucination risk. **The hard part is deciding *when* to trigger it and *which* facts to feed it**, and that turns out to be a harder version of the same problem retrieval already struggles with:

- Keyword-matching the question ("délai", "dans les temps", "à temps") is brittle — it misses indirect phrasings and doesn't generalize to other deadline-shaped questions (vol: 2 jours, catastrophe naturelle: 30 jours) without a growing, hand-maintained pattern list.
- Even a correctly detected "deadline question" still needs to resolve *which entity* it's about and *which two dates, from which specific document* should feed the computation — which is the retrieval/attribution problem all over again, just relocated to a sub-task instead of solved.

**Decision: move structured fact extraction to ingestion time instead of query time.** Rather than deciding at question-time whether to verify, extract `loss_date` and `declaration_date` for each fiche sinistre when the corpus is built, store them as queryable fields (extending the same separation-of-concerns pattern already used for `metadata.json`), and make deadline-compliance a deterministic database lookup — independent of how the question happens to be phrased. This sidesteps the trigger-detection problem entirely instead of trying to make it more robust.

---

## Jour 3, Part 4: A Deterministic Check, Built Specifically to Catch the LLM Being Wrong

**Built:** `claim_facts.json` holds hand-extracted structured fields per sinistre (`entity_name`, `loss_date`, `declaration_date`, `deadline_rule`) — same separation-of-concerns spirit as `metadata.json`, but for per-claim facts instead of per-document tags. `claim_verification.py` computes business-day gaps with `numpy.busday_count` and checks them against the applicable rule, entirely independent of the LLM. `rag.py` detects a known entity name in the question (a deliberately simple substring match — not meant to replace retrieval, only to trigger a side-channel check alongside it) and prints the deterministic result directly under the LLM's answer.

**This wasn't built on a hunch — it was built because Jour 3 Parts 2–3 had just shown that "valid citation" and "correct conclusion" are two different, separable properties.** The deterministic layer exists specifically to test that gap with a real comparison, not to add a feature for its own sake.

**Result, run immediately after wiring it in:**

| Question | LLM answer | Deterministic check |
|---|---|---|
| Martin dans les temps ? | "Oui... [A]" | **CONFORME** — 3 jours ouvrés (limite : 5) |
| Lefèvre dans les temps ? | "Oui... ([A])" | **NON CONFORME** — 10 jours ouvrés (limite : 5) |

**Martin: agreement.** **Lefèvre: direct contradiction** — the LLM said "Oui," citing its own correctly-attributed source document, while the deterministic computation against that same document's dates says "Non." This is the cleanest possible proof of the Jour 3 Part 3 finding: a citation can be entirely valid and the conclusion built on it can still be wrong, and no amount of better prompting catches that, because the LLM was never actually performing the comparison — it was pattern-matching "Oui" + a citation tag, not computing 15/02/2025 − 03/02/2025 and checking it against a threshold.

**Why this result matters more than "the fix worked":** it's not just that the deterministic layer agrees with itself (trivially true — it's arithmetic). It's that **it was used to catch a live disagreement with the LLM, on the exact case engineered to be ambiguous enough to fool free-form generation.** That's the actual validation of the architectural decision made in "Why Not Just Detect Deadline Questions" above: removing the LLM from the arithmetic entirely, rather than trying to prompt it into doing arithmetic reliably, is what makes the check trustworthy.

---

## Conclusion: Three Days, One Recurring Lesson

Each day of this project assumed something was fixed, then found the next layer where the same underlying problem reappeared in a sharper form:

- **Jour 1** assumed naive retrieval was the problem. It was — but fixing lexical similarity (Jour 2 metadata filtering) didn't fix chunk boundaries cutting through logic, which didn't fix non-deterministic generation, which didn't fix cross-document contamination, which didn't fix unverified citations.
- **Every fix was real and verified** — each one demonstrably solved the specific failure it targeted, with before/after evidence, not just a claim. None of them were wasted work.
- **And every fix also relocated the underlying problem one layer deeper**, rather than eliminating it: from "wrong document retrieved" → "right document, wrong chunk" → "right chunk, wrong reasoning" → "right reasoning shown, but not actually performed."

**The final, most general lesson:** in a regulated domain, the question "did the system get the right answer?" is not the same question as "did the system perform the operation that justifies that answer?" A RAG pipeline can pass every individual-component check — correct retrieval, correct citation, plausible-sounding generation — and still be wrong at the one step that actually matters, because nothing in a standard RAG pipeline forces verification of the reasoning step itself. The only fix that closed this gap completely was removing the LLM from that specific operation (date arithmetic) and replacing it with deterministic code — not better prompting, not more context, not smarter retrieval.

**This generalizes beyond dates.** Anywhere a question requires a *computation* or *comparison* — premium calculations, coverage limit checks, multi-document consistency checks — the same pattern likely applies: retrieval and generation can get arbitrarily good at finding and presenting the right facts, while still failing at the step of actually reasoning over them correctly. The practical implication for a tool like this in production: identify which sub-tasks are computations in disguise, and route those to deterministic code, rather than trusting a language model to both find and compute correctly just because it can do each separately under easy conditions.

---

## Known Limitations

**Resolved in Jour 2:**
- ~~No metadata filtering~~ → fixed via `metadata.json` + SQL filter (see above), though filter selection is still manual, not automatic
- ~~Chunk boundaries severing numeric thresholds from explanations~~ → fixed via structure-aware chunking (see above)
- ~~Non-deterministic generation (contradictory answers on identical input)~~ → fixed via `temperature=0` (see Jour 2, Part 3)

**Resolved in Jour 3:**
- ~~Cross-document fact contamination (citing one person's date for another's claim)~~ → fixed via prompt-level per-fact citation (see Jour 3, Part 2)
- ~~Two related facts (loss date, declaration date) split across different chunks of the same short document~~ → fixed via whole-document chunking for short texts (see Jour 3, Part 3)
- ~~No verification that a cited fact supports the stated conclusion~~ → fixed via a deterministic side-channel (`claim_verification.py`) that recomputes deadline compliance independently of the LLM — see Jour 3, Part 4, including a real case where it caught the LLM giving a confidently wrong answer

**Still open:**
- **The deterministic check runs alongside the LLM's answer, not instead of it.** Both are shown to the user; nothing yet decides which one to trust when they disagree, or surfaces the disagreement as a flagged warning rather than two unlabeled blocks of text. A user who only reads the LLM's fluent answer can still miss the contradiction sitting right below it.
- **Entity detection is a simple substring match**, not real entity resolution — works for this corpus's distinct names, would need real disambiguation (e.g. matching contract numbers, not just last names) at any meaningful scale.
- **`claim_facts.json` is hand-extracted**, the same manual-curation tradeoff already accepted for `metadata.json`. A real system would need an extraction pipeline (regex, structured parsing, or a constrained LLM extraction step with its own verification) rather than hand-typed JSON.
- **No automatic filter selection:** The hybrid retrieval filter (`contrat:xxx`, `type:xxx`) has to be typed explicitly by the user.
- **Single-turn retrieval:** No iterative refinement. If top-k misses the right answer, the model can't ask for more information.
- **No re-embedding cache:** Every run re-embeds the entire corpus from scratch, even if nothing changed. Parked as a nice-to-have, not core to the RAG-quality story.

---

## What's Left

1. **Make disagreement between the LLM and the deterministic check impossible to miss** — e.g. an explicit warning line when the two verdicts differ, rather than two side-by-side blocks the user has to compare themselves.
2. **A real evaluation set** — curated Q&A pairs with known ground-truth answers, so future changes can be checked against a fixed benchmark instead of re-running the same handful of manual questions each time.
3. **Automatic filter inference:** Classify the question's contract type automatically instead of requiring manual `contrat:xxx` prefixes.
4. **A real extraction pipeline for `claim_facts.json`**, replacing hand-typed entries with something that scales past a handful of test documents.
5. **Possibly:** expose this RAG via a minimal MCP server, once grounding is solid enough to trust in an agent context.

---

## Corpus

Real documents from insurers (public CGs, actual Code des assurances extracts, internal procedure notes). Not synthetic, which means extraction noise, mixed contract types, and real-world messiness — exactly the challenges a junior consultant will face.

Skip list:
- `Code des assurances.pdf` (full 12MB version) — too large; using targeted extracts instead
- `conditions_generales.txt` (empty file, left by mistake)

---

## Running It

```bash
# one-time setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# day-to-day
python3 rag.py
```

Ask questions. Watch the retrieved chunks. Notice where the model's confidence exceeds what the data actually supports — and notice when a confident-sounding citation doesn't actually belong to the document or person being discussed. That's the whole exercise, across all three days.

**Optional inline filters**, now that hybrid retrieval is in:
```
contrat:dommages_eaux Quel est le délai de déclaration ?
type:procedure_interne Quel est le seuil de dispense d'expertise ?
```

**Deterministic deadline check:** if a question mentions a name present in `claim_facts.json` (currently: Dupont, Martin, Lefevre — see that file to add more), a second block appears automatically below the LLM's answer:

```
🧮 Vérification déterministe (claim_facts.json) pour 'Lefevre' :
   [Sinistre Lefevre hors delais.odt] NON CONFORME — 10 jour(s) ouvré(s) entre la survenance et la déclaration (limite : 5 jours ouvrés, hors jours fériés non pris en compte).
```

This check never touches the LLM — it's a plain Python lookup and `numpy.busday_count` calculation against hand-extracted dates. Compare it against the LLM's answer above it; they won't always agree, and that disagreement is itself the most important finding in this README (see Jour 3, Part 4 and the Conclusion).
