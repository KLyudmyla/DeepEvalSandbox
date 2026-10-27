"""Layer 2 document-retrieval golden dataset - Stratified & Expanded Benchmark."""

from __future__ import annotations

from typing import Any

# Corpus reference filenames:
# - engineering_stack.txt
# - hr_remote_work.txt
# - it_laptop_request.txt
# - office_kitchen.txt
# - valid_document.txt

GOLDEN_DATASET: list[dict[str, Any]] = [
    # =========================================================================
    # ORIGINAL CASES (1-5)
    # =========================================================================
    {
        "id": "engineering-preferred-framework",
        "query": "Which framework is preferred for new Python APIs?",
        "ground_truth_document_ids": ["engineering_stack.txt"],
        "expected_metadata_filters": {"source": "engineering_stack.txt"},
        "graded_relevance_scores": {"engineering_stack.txt": 3},
        "k": 2,
        "tags": ["single_doc", "fact_retrieval", "engineering", "easy", "en"],
        "query_type": "answerable"
    },
    {
        "id": "anniversary-bonus-15-years",
        "query": "What is the amount of bonus for a 15 year company anniversary?",
        "ground_truth_document_ids": ["valid_document.txt"],
        "expected_metadata_filters": {"source": "valid_document.txt"},
        "graded_relevance_scores": {"valid_document.txt": 3},
        "k": 2,
        "tags": ["single_doc", "amount_retrieval", "compensation", "easy", "en"],
        "query_type": "answerable"
    },
    {
        "id": "legalization-spain",
        "query": "Who can help with legalization issues in Spain?",
        "ground_truth_document_ids": ["valid_document.txt"],
        "expected_metadata_filters": {"source": "valid_document.txt"},
        "graded_relevance_scores": {"valid_document.txt": 3},
        "k": 2,
        "tags": ["single_doc", "contact_retrieval", "legalization", "easy", "en"],
        "query_type": "answerable"
    },
    {
        "id": "remote-work-policy",
        "query": "What is the company policy for working remotely?",
        "ground_truth_document_ids": ["hr_remote_work.txt"],
        "expected_metadata_filters": {"source": "hr_remote_work.txt"},
        "graded_relevance_scores": {"hr_remote_work.txt": 3},
        "k": 2,
        "tags": ["single_doc", "policy", "hr", "easy", "en"],
        "query_type": "answerable"
    },
    {
        "id": "unknown-germany-legalization",
        "query": "Who is the legalization specialist for Germany?",
        "ground_truth_document_ids": [],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {},
        "k": 2,
        "tags": ["abstention", "no_positive_evidence", "easy", "en"],
        "query_type": "unanswerable"
    },

    # =========================================================================
    # STRATUM 1: MULTI-DOCUMENT SYNTHESIS & CROSS-CORPUS RETRIEVAL (6-13)
    # =========================================================================
    {
        "id": "multi-remote-it-equipment",
        "query": "What are the rules regarding company hardware when working remotely?",
        "ground_truth_document_ids": ["hr_remote_work.txt", "it_laptop_request.txt"],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {"hr_remote_work.txt": 3, "it_laptop_request.txt": 3},
        "k": 3,
        "tags": ["multi_doc", "policy_synthesis", "hr", "it", "medium", "en"],
        "query_type": "answerable"
    },
    {
        "id": "multi-contractor-overtime-and-equipment",
        "query": "What rules apply to B2B contractors regarding overtime limits and laptop requests?",
        "ground_truth_document_ids": ["valid_document.txt", "it_laptop_request.txt"],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {"valid_document.txt": 3, "it_laptop_request.txt": 2},
        "k": 3,
        "tags": ["multi_doc", "contractor_rules", "compensation", "it", "medium", "en"],
        "query_type": "answerable"
    },
    {
        "id": "multi-office-conduct-kitchen-remote",
        "query": "What expectations exist for office presence, physical attendance, and kitchen etiquette?",
        "ground_truth_document_ids": ["hr_remote_work.txt", "office_kitchen.txt"],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {"hr_remote_work.txt": 3, "office_kitchen.txt": 3},
        "k": 3,
        "tags": ["multi_doc", "office_rules", "hr", "admin", "medium", "en"],
        "query_type": "answerable"
    },
    {
        "id": "multi-tech-and-it-procurement",
        "query": "What software stack is approved for backend engineering and how do engineers get specialized hardware?",
        "ground_truth_document_ids": ["engineering_stack.txt", "it_laptop_request.txt"],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {"engineering_stack.txt": 3, "it_laptop_request.txt": 3},
        "k": 3,
        "tags": ["multi_doc", "engineering_it", "engineering", "it", "hard", "en"],
        "query_type": "answerable"
    },
    {
        "id": "multi-perks-bonuses-and-reimbursements",
        "query": "List all cash bonuses, wellness allowances, and equipment policies provided by the company.",
        "ground_truth_document_ids": ["valid_document.txt", "it_laptop_request.txt"],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {"valid_document.txt": 3, "it_laptop_request.txt": 2},
        "k": 4,
        "tags": ["multi_doc", "benefits_summary", "compensation", "it", "hard", "en"],
        "query_type": "answerable"
    },
    {
        "id": "multi-hr-comprehensive-policy",
        "query": "What policies cover employee leaves, remote work from abroad, and anniversary recognition?",
        "ground_truth_document_ids": ["hr_remote_work.txt", "valid_document.txt"],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {"hr_remote_work.txt": 3, "valid_document.txt": 3},
        "k": 3,
        "tags": ["multi_doc", "hr_policy", "hr", "compensation", "medium", "en"],
        "query_type": "answerable"
    },
    {
        "id": "multi-workplace-support-contacts",
        "query": "Who should employees contact for tech questions, equipment requests, kitchen breakage, or Spanish legalization?",
        "ground_truth_document_ids": [
            "engineering_stack.txt",
            "it_laptop_request.txt",
            "office_kitchen.txt",
            "valid_document.txt",
        ],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {
            "engineering_stack.txt": 2,
            "it_laptop_request.txt": 2,
            "office_kitchen.txt": 2,
            "valid_document.txt": 3,
        },
        "k": 5,
        "tags": ["multi_doc", "support_routing", "cross_department", "hard", "en"],
        "query_type": "answerable"
    },
    {
        "id": "multi-b2b-vs-employee-conditions",
        "query": "What working hours, overtime caps, and hardware ownership terms apply to company staff vs contractors?",
        "ground_truth_document_ids": ["valid_document.txt", "hr_remote_work.txt", "it_laptop_request.txt"],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {
            "valid_document.txt": 3,
            "hr_remote_work.txt": 2,
            "it_laptop_request.txt": 2,
        },
        "k": 4,
        "tags": ["multi_doc", "employment_terms", "compensation", "hr", "hard", "en"],
        "query_type": "answerable"
    },

    # =========================================================================
    # STRATUM 2: PARAPHRASES, SYNONYMS, AND IMPLICIT INTENT (14-21)
    # =========================================================================
    {
        "id": "para-fridge-cleaning-time",
        "query": "When are items thrown away or cleared out from the shared refrigerator?",
        "ground_truth_document_ids": ["office_kitchen.txt"],
        "expected_metadata_filters": {"source": "office_kitchen.txt"},
        "graded_relevance_scores": {"office_kitchen.txt": 3},
        "k": 2,
        "tags": ["paraphrase", "office_kitchen", "medium", "en"],
        "query_type": "answerable"
    },
    {
        "id": "para-gym-fitness-stipend",
        "query": "How much money can I get reimbursed for my monthly gym membership?",
        "ground_truth_document_ids": ["valid_document.txt"],
        "expected_metadata_filters": {"source": "valid_document.txt"},
        "graded_relevance_scores": {"valid_document.txt": 3, "hr_remote_work.txt": 1},
        "k": 2,
        "tags": ["paraphrase", "compensation", "medium", "en"],
        "query_type": "answerable"
    },
    {
        "id": "para-new-parent-paternity-maternity",
        "query": "Is there financial support or extra salary given for welcoming a new child through adoption or birth?",
        "ground_truth_document_ids": ["valid_document.txt"],
        "expected_metadata_filters": {"source": "valid_document.txt"},
        "graded_relevance_scores": {"valid_document.txt": 3},
        "k": 2,
        "tags": ["paraphrase", "compensation", "medium", "en"],
        "query_type": "answerable"
    },
    {
        "id": "para-django-python-framework",
        "query": "Can I use Django to build a brand new REST API for the backend?",
        "ground_truth_document_ids": ["engineering_stack.txt"],
        "expected_metadata_filters": {"source": "engineering_stack.txt"},
        "graded_relevance_scores": {"engineering_stack.txt": 3},
        "k": 2,
        "tags": ["paraphrase", "engineering", "medium", "en"],
        "query_type": "answerable"
    },
    {
        "id": "para-laptop-replacement-form",
        "query": "Which document or request form do I need to fill out if my computer is broken?",
        "ground_truth_document_ids": ["it_laptop_request.txt"],
        "expected_metadata_filters": {"source": "it_laptop_request.txt"},
        "graded_relevance_scores": {"it_laptop_request.txt": 3},
        "k": 2,
        "tags": ["paraphrase", "it", "medium", "en"],
        "query_type": "answerable"
    },
    {
        "id": "para-work-from-home-allowance",
        "query": "How many days a week am I permitted to telecommute from my house?",
        "ground_truth_document_ids": ["hr_remote_work.txt"],
        "expected_metadata_filters": {"source": "hr_remote_work.txt"},
        "graded_relevance_scores": {"hr_remote_work.txt": 3},
        "k": 2,
        "tags": ["paraphrase", "hr", "medium", "en"],
        "query_type": "answerable"
    },
    {
        "id": "para-frontend-web-tech",
        "query": "What client-side JavaScript library or UI framework does the company standardise on?",
        "ground_truth_document_ids": ["engineering_stack.txt"],
        "expected_metadata_filters": {"source": "engineering_stack.txt"},
        "graded_relevance_scores": {"engineering_stack.txt": 3},
        "k": 2,
        "tags": ["paraphrase", "engineering", "hard", "en"],
        "query_type": "answerable"
    },
    {
        "id": "para-third-party-gym-payment",
        "query": "Will the business pay for my gym ticket if my spouse paid for it from their account?",
        "ground_truth_document_ids": ["valid_document.txt"],
        "expected_metadata_filters": {"source": "valid_document.txt"},
        "graded_relevance_scores": {"valid_document.txt": 3},
        "k": 2,
        "tags": ["paraphrase", "implicit_rule", "compensation", "hard", "en"],
        "query_type": "answerable"
    },

    # =========================================================================
    # STRATUM 3: NUMERICAL, CONSTRAINTS & FINE-GRAINED DRILL-DOWN (22-28)
    # =========================================================================
    {
        "id": "num-overtime-cap-contractors",
        "query": "What is the maximum allowed weekly overtime for B2B contractors vs contract employees?",
        "ground_truth_document_ids": ["valid_document.txt"],
        "expected_metadata_filters": {"source": "valid_document.txt"},
        "graded_relevance_scores": {"valid_document.txt": 3},
        "k": 2,
        "tags": ["numerical", "compensation", "medium", "en"],
        "query_type": "answerable"
    },
    {
        "id": "num-anniversary-bonus-20-years",
        "query": "How much bonus is granted for reaching 20 years of continuous service?",
        "ground_truth_document_ids": ["valid_document.txt"],
        "expected_metadata_filters": {"source": "valid_document.txt"},
        "graded_relevance_scores": {"valid_document.txt": 3},
        "k": 2,
        "tags": ["numerical", "compensation", "easy", "en"],
        "query_type": "answerable"
    },
    {
        "id": "num-anniversary-tier-milestones",
        "query": "List all tenure milestone years and their corresponding dollar bonus payouts.",
        "ground_truth_document_ids": ["valid_document.txt"],
        "expected_metadata_filters": {"source": "valid_document.txt"},
        "graded_relevance_scores": {"valid_document.txt": 3},
        "k": 2,
        "tags": ["numerical", "compensation", "medium", "en"],
        "query_type": "answerable"
    },
    {
        "id": "num-kitchen-cleaning-cutoff-time",
        "query": "At what specific time and day of the week is the office kitchen cleared?",
        "ground_truth_document_ids": ["office_kitchen.txt"],
        "expected_metadata_filters": {"source": "office_kitchen.txt"},
        "graded_relevance_scores": {"office_kitchen.txt": 3},
        "k": 2,
        "tags": ["numerical", "office_kitchen", "easy", "en"],
        "query_type": "answerable"
    },
    {
        "id": "num-max-remote-days-per-week",
        "query": "What is the upper limit on weekly remote days permitted under standard policy?",
        "ground_truth_document_ids": ["hr_remote_work.txt"],
        "expected_metadata_filters": {"source": "hr_remote_work.txt"},
        "graded_relevance_scores": {"hr_remote_work.txt": 3},
        "k": 2,
        "tags": ["numerical", "hr", "easy", "en"],
        "query_type": "answerable"
    },
    {
        "id": "num-gym-reimbursement-cap",
        "query": "What is the maximum monthly limit for sports and gym reimbursement?",
        "ground_truth_document_ids": ["valid_document.txt"],
        "expected_metadata_filters": {"source": "valid_document.txt"},
        "graded_relevance_scores": {"valid_document.txt": 3},
        "k": 2,
        "tags": ["numerical", "compensation", "easy", "en"],
        "query_type": "answerable"
    },
    {
        "id": "num-form-code-laptop-request",
        "query": "What exact form code must be submitted to request a replacement laptop?",
        "ground_truth_document_ids": ["it_laptop_request.txt"],
        "expected_metadata_filters": {"source": "it_laptop_request.txt"},
        "graded_relevance_scores": {"it_laptop_request.txt": 3},
        "k": 2,
        "tags": ["numerical", "entity_lookup", "it", "easy", "en"],
        "query_type": "answerable"
    },

    # =========================================================================
    # STRATUM 4: MULTILINGUAL EVALUATION (SPANISH & POLISH) (29-34)
    # =========================================================================
    {
        "id": "lang-es-legalization-contact",
        "query": "¿Quién es la persona de contacto para los trámites de legalización en España y cuál es su correo?",
        "ground_truth_document_ids": ["valid_document.txt"],
        "expected_metadata_filters": {"source": "valid_document.txt"},
        "graded_relevance_scores": {"valid_document.txt": 3},
        "k": 2,
        "tags": ["multilingual", "spanish", "legalization", "medium", "es"],
        "query_type": "answerable"
    },
    {
        "id": "lang-es-remote-work-days",
        "query": "¿Cuántos días a la semana se permite trabajar desde casa según la política de la empresa?",
        "ground_truth_document_ids": ["hr_remote_work.txt"],
        "expected_metadata_filters": {"source": "hr_remote_work.txt"},
        "graded_relevance_scores": {"hr_remote_work.txt": 3},
        "k": 2,
        "tags": ["multilingual", "spanish", "hr", "medium", "es"],
        "query_type": "answerable"
    },
    {
        "id": "lang-es-kitchen-cleaning-rule",
        "query": "¿Cuándo se limpia el refrigerador de la oficina y qué pasa con la comida sin etiquetar?",
        "ground_truth_document_ids": ["office_kitchen.txt"],
        "expected_metadata_filters": {"source": "office_kitchen.txt"},
        "graded_relevance_scores": {"office_kitchen.txt": 3},
        "k": 2,
        "tags": ["multilingual", "spanish", "office_kitchen", "medium", "es"],
        "query_type": "answerable"
    },
    {
        "id": "lang-pl-python-framework",
        "query": "Jaki framework jest zalecany do tworzenia nowych API w języku Python?",
        "ground_truth_document_ids": ["engineering_stack.txt"],
        "expected_metadata_filters": {"source": "engineering_stack.txt"},
        "graded_relevance_scores": {"engineering_stack.txt": 3},
        "k": 2,
        "tags": ["multilingual", "polish", "engineering", "medium", "pl"],
        "query_type": "answerable"
    },
    {
        "id": "lang-pl-laptop-request-procedure",
        "query": "Jakie są procedury wnioskowania o nowy komputer służbowy?",
        "ground_truth_document_ids": ["it_laptop_request.txt"],
        "expected_metadata_filters": {"source": "it_laptop_request.txt"},
        "graded_relevance_scores": {"it_laptop_request.txt": 3},
        "k": 2,
        "tags": ["multilingual", "polish", "it", "medium", "pl"],
        "query_type": "answerable"
    },
    {
        "id": "lang-pl-gym-reimbursement",
        "query": "Jaka jest maksymalna kwota zwrotu za siłownię i jakie są wymagania dotyczące płatności?",
        "ground_truth_document_ids": ["valid_document.txt"],
        "expected_metadata_filters": {"source": "valid_document.txt"},
        "graded_relevance_scores": {"valid_document.txt": 3},
        "k": 2,
        "tags": ["multilingual", "polish", "compensation", "hard", "pl"],
        "query_type": "answerable"
    },

    # =========================================================================
    # STRATUM 5: ABSTENTION, OUT-OF-BOUNDS & NEAR-MISS NEGATIVES (35-39)
    # =========================================================================
    {
        "id": "abs-unsupported-car-allowance",
        "query": "What is the company policy for company car allowance and mileage reimbursement?",
        "ground_truth_document_ids": [],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {},
        "k": 2,
        "tags": ["abstention", "out_of_bounds", "easy", "en"],
        "query_type": "unanswerable"
    },
    {
        "id": "abs-unsupported-relocation-uk",
        "query": "Who is the HR contact responsible for visa legalization in the United Kingdom?",
        "ground_truth_document_ids": [],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {"valid_document.txt": 1},  # Distractor near-miss
        "k": 2,
        "tags": ["abstention", "near_miss_distractor", "hard", "en"],
        "query_type": "unanswerable"
    },
    {
        "id": "abs-unsupported-flask-support",
        "query": "Is Flask explicitly listed as the standard microframework for backend microservices?",
        "ground_truth_document_ids": [],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {"engineering_stack.txt": 1},  # Engineering distractor
        "k": 2,
        "tags": ["abstention", "engineering_distractor", "medium", "en"],
        "query_type": "unanswerable"
    },
    {
        "id": "abs-unsupported-pet-policy",
        "query": "Are dogs or other pets allowed inside the office space during regular working hours?",
        "ground_truth_document_ids": [],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {"office_kitchen.txt": 1},
        "k": 2,
        "tags": ["abstention", "out_of_bounds", "easy", "en"],
        "query_type": "unanswerable"
    },
    {
        "id": "abs-unsupported-macbook-pro",
        "query": "Can employees buy a custom Apple M3 Max MacBook Pro directly and expense it on their corporate credit card?",
        "ground_truth_document_ids": ["it_laptop_request.txt"],
        "expected_metadata_filters": {"source": "it_laptop_request.txt"},
        "graded_relevance_scores": {"it_laptop_request.txt": 3},  # Explicitly answered as disallowed in doc
        "k": 2,
        "tags": ["negative_rule_retrieval", "it", "hard", "en"],
        "query_type": "answerable"
    },

    # =========================================================================
    # STRATUM 6: NOISY QUERIES, TYPOS & KNOWN FAILURE MODES (40-45)
    # =========================================================================
    {
        "id": "noise-typos-fastapi-react",
        "query": "Which framwork is prefered for Pythn APIs and frontend web app development?",
        "ground_truth_document_ids": ["engineering_stack.txt"],
        "expected_metadata_filters": {"source": "engineering_stack.txt"},
        "graded_relevance_scores": {"engineering_stack.txt": 3},
        "k": 2,
        "tags": ["typos", "robustness", "engineering", "medium", "en"],
        "query_type": "answerable"
    },
    {
        "id": "noise-typos-anniversary-bonus",
        "query": "Waht is the bonuss for 10 yars of continuos service at the compny?",
        "ground_truth_document_ids": ["valid_document.txt"],
        "expected_metadata_filters": {"source": "valid_document.txt"},
        "graded_relevance_scores": {"valid_document.txt": 3},
        "k": 2,
        "tags": ["typos", "robustness", "compensation", "medium", "en"],
        "query_type": "answerable"
    },
    {
        "id": "distractor-laptop-vs-kitchen-cleaning",
        "query": "Are employees allowed to leave personal equipment or laptops in the kitchen overnight?",
        "ground_truth_document_ids": ["office_kitchen.txt"],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {"office_kitchen.txt": 3, "it_laptop_request.txt": 1},
        "k": 3,
        "tags": ["distractor_trap", "multi_doc", "hard", "en"],
        "query_type": "answerable"
    },
    {
        "id": "distractor-sick-leave-vs-remote-work",
        "query": "Can an employee who is sick work remotely from home instead of taking sick leave?",
        "ground_truth_document_ids": ["hr_remote_work.txt"],
        "expected_metadata_filters": {"source": "hr_remote_work.txt"},
        "graded_relevance_scores": {"hr_remote_work.txt": 3},
        "k": 2,
        "tags": ["policy_trap", "hr", "hard", "en"],
        "query_type": "answerable"
    },
    {
        "id": "meta-conflict-source-override",
        "query": "Where do I report broken equipment according to office rules?",
        "ground_truth_document_ids": ["office_kitchen.txt"],
        "expected_metadata_filters": {"source": "office_kitchen.txt"},
        "graded_relevance_scores": {"office_kitchen.txt": 3, "it_laptop_request.txt": 1},
        "k": 2,
        "tags": ["metadata_filter", "admin", "medium", "en"],
        "query_type": "answerable"
    },
    {
        "id": "distractor-foreign-remote-work-vs-legalization",
        "query": "Does Sara Parker handle approvals for working remotely from another country?",
        "ground_truth_document_ids": ["hr_remote_work.txt", "valid_document.txt"],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {"hr_remote_work.txt": 3, "valid_document.txt": 2},
        "k": 3,
        "tags": ["cross_cutting_distractor", "hr", "legalization", "very_hard", "en"],
        "query_type": "answerable"
    },
]

# Preserve original export names.
MY_DATASET = GOLDEN_DATASET

__all__ = ["GOLDEN_DATASET", "MY_DATASET"]