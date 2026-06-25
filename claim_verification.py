"""
Vérification déterministe des délais de déclaration, à partir de faits
structurés extraits à l'ingestion (claim_facts.json) plutôt que de faits
trouvés et comparés par le LLM à la volée.

Pourquoi ce module existe (voir README, Jour 3) :
le LLM, même avec citations correctes, peut affirmer un verdict de délai
sans réellement calculer l'écart entre date de survenance et date de
déclaration. Ce module retire complètement le LLM de cette opération :
les dates sont extraites une fois, à la main ou via un futur pipeline
d'extraction, et le calcul de conformité est un calcul Python pur,
vérifiable indépendamment de toute génération de texte.

Limite assumée : numpy.busday_count exclut les week-ends mais pas les
jours fériés français — une approximation suffisante pour ce prototype,
documentée plutôt que masquée.
"""

import json
import numpy as np
from datetime import date


def load_claim_facts(path: str = "claim_facts.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_claim_by_entity(entity_name_fragment: str, facts: dict) -> list[dict]:
    """
    Cherche les sinistres correspondant à un nom (recherche insensible à la
    casse, par sous-chaîne — ex: 'Lefèvre' ou 'Lefevre' doit matcher
    'Bernard Lefevre'). Renvoie une LISTE : un même nom peut correspondre à
    plusieurs sinistres (cf. Dupont, qui a deux dossiers distincts) — c'est
    précisément le cas que le LLM confondait avant la correction des prompts.
    """
    normalized = (
        entity_name_fragment.lower()
        .replace("è", "e").replace("é", "e").replace("ê", "e")
    )
    matches = []
    for claim in facts["claims"]:
        claim_name_normalized = (
            claim["entity_name"].lower()
            .replace("è", "e").replace("é", "e").replace("ê", "e")
        )
        if normalized in claim_name_normalized or any(
            part in claim_name_normalized for part in normalized.split()
        ):
            matches.append(claim)
    return matches


def check_deadline_compliance(claim: dict, facts: dict) -> dict:
    """
    Calcule si une déclaration a été faite dans le délai applicable.
    Renvoie un dict avec le résultat ET les valeurs utilisées, pour que le
    calcul reste entièrement auditable (pas de boîte noire).
    """
    if not claim.get("loss_date") or not claim.get("declaration_date"):
        return {
            "compliant": None,
            "reason": "Date de survenance ou de déclaration manquante dans claim_facts.json — "
                      "vérification impossible, pas de réponse par défaut.",
        }

    loss_date = date.fromisoformat(claim["loss_date"])
    declaration_date = date.fromisoformat(claim["declaration_date"])

    rule_key = claim.get("deadline_rule")
    rule = facts.get("deadline_rules", {}).get(rule_key)
    if not rule:
        return {
            "compliant": None,
            "reason": f"Règle de délai '{rule_key}' introuvable dans claim_facts.json.",
        }

    business_days_elapsed = int(
        np.busday_count(np.datetime64(loss_date), np.datetime64(declaration_date))
    )
    max_days = rule["max_business_days"]
    compliant = business_days_elapsed <= max_days

    return {
        "compliant": compliant,
        "entity_name": claim["entity_name"],
        "loss_date": claim["loss_date"],
        "declaration_date": claim["declaration_date"],
        "business_days_elapsed": business_days_elapsed,
        "max_business_days_allowed": max_days,
        "rule_source": rule["source"],
        "reason": (
            f"{business_days_elapsed} jour(s) ouvré(s) entre la survenance et la déclaration "
            f"(limite : {max_days} jours ouvrés, hors jours fériés non pris en compte)."
        ),
    }


def verify_entity_deadline(entity_name_fragment: str, facts_path: str = "claim_facts.json") -> list[dict]:
    """
    Point d'entrée principal : cherche tous les sinistres correspondant à un
    nom, et calcule la conformité de délai pour chacun. Renvoie une liste
    (jamais un seul résultat implicite) pour rendre visible, dès la sortie
    de la fonction, le cas où un même nom a plusieurs dossiers — au lieu de
    masquer l'ambiguïté comme le faisait la génération LLM.
    """
    facts = load_claim_facts(facts_path)
    matches = find_claim_by_entity(entity_name_fragment, facts)

    if not matches:
        return [{
            "compliant": None,
            "reason": f"Aucun sinistre trouvé pour '{entity_name_fragment}' dans claim_facts.json.",
        }]

    results = []
    for claim in matches:
        result = check_deadline_compliance(claim, facts)
        result["matched_filename"] = claim["filename"]
        result["claim_type"] = claim.get("claim_type")
        results.append(result)
    return results


if __name__ == "__main__":
    # Auto-test rapide sur les 4 cas connus du corpus, pour vérifier la
    # logique indépendamment de toute intégration avec rag.py.
    test_names = ["Dupont", "Martin", "Lefevre", "Lefèvre", "Personne Inconnue"]
    for name in test_names:
        print(f"--- Recherche : '{name}' ---")
        results = verify_entity_deadline(name)
        for r in results:
            print(f"  {r}")
        print()
