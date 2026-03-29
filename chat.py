"""chat.py

Pipeline:
1) User text -> BN -> top diseases
2) Pick best disease file: intent<disease>.json
3) Detect user question type: definition / cause / symptom / prevention / food / medicine / post-recovery
4) TF-IDF match only within that tag
5) Return the best intent response
6) Keep disease context for follow-up questions on the same disease

This version is deterministic and does NOT use GPT-2.

Install:
    pip install scikit-learn
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from inpToBn import statement_to_bn_input
from decision import find_top_k


# =========================
# helpers
# =========================

TAG_ORDER = [
    "definition",
    "cause",
    "symptom",
    "prevention",
    "food",
    "medicine",
    "post-recovery",
]

TAG_KEYWORDS = {
    "definition": [
        "what is",
        "define",
        "meaning",
        "about",
        "explain",
        "description",
        "details",
        "is it",
    ],
    "cause": [
        "why",
        "cause",
        "causes",
        "reason",
        "reason for",
        "trigger",
        "because of",
        "due to",
    ],
    "symptom": [
        "symptom",
        "symptoms",
        "feel",
        "feeling",
        "having",
        "experience",
        "experiencing",
        "what are the signs",
        "signs",
    ],
    "prevention": [
        "prevent",
        "prevention",
        "avoid",
        "protect",
        "stop",
        "reduce chance",
        "how can i avoid",
    ],
    "food": [
        "food",
        "eat",
        "eating",
        "diet",
        "what to eat",
        "what should i eat",
        "can i eat",
        "avoid food",
        "nutrition",
    ],
    "medicine": [
        "medicine",
        "medication",
        "drug",
        "tablet",
        "pill",
        "treatment",
        "treat",
        "cure",
        "prescribe",
        "dose",
        "dosage",
    ],
    "post-recovery": [
        "after recovery",
        "recover",
        "recovery",
        "after getting better",
        "after flu",
        "after treatment",
        "post recovery",
        "what next",
    ],
}

FOLLOWUP_QUESTION_TEMPLATES = {
    "definition": [
        "What are the common symptoms?",
        "What causes this condition?",
        "How can it be prevented?",
    ],
    "cause": [
        "What are the symptoms?",
        "How is it prevented?",
        "What medicine is usually used?",
    ],
    "symptom": [
        "What is the condition?",
        "What causes it?",
        "How can it be treated or prevented?",
    ],
    "prevention": [
        "What are the symptoms?",
        "What causes it?",
        "What should I eat or avoid?",
    ],
    "food": [
        "What foods should I avoid?",
        "What are the symptoms?",
        "What medicine is usually used?",
    ],
    "medicine": [
        "What are the symptoms?",
        "What causes it?",
        "What should I do after recovery?",
    ],
    "post-recovery": [
        "What food is good during recovery?",
        "How can I prevent it again?",
        "What symptoms should I watch for?",
    ],
}


def norm(s: str) -> str:
    """Lowercase and remove punctuation/spaces for safe matching."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def norm_keep_spaces(s: str) -> str:
    """Lowercase and keep spaces for TF-IDF text matching."""
    s = s.lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_topk(top_k: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for x in top_k:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, (list, tuple)) and x:
            out.append(str(x[0]))
        else:
            out.append(str(x))
    return out


# =========================
# models
# =========================


@dataclass
class IntentItem:
    tag: str
    patterns: List[str]
    responses: List[str]


@dataclass
class IntentFile:
    disease_key: str
    filename: str
    display_name: str
    intents: List[IntentItem]


# =========================
# intent loading
# =========================


class IntentStore:
    def __init__(self, path: str = "intents"):
        self.path = Path(path)
        self.files: Dict[str, IntentFile] = {}
        self._load()

    def _file_key(self, stem_without_intent: str) -> str:
        return norm(stem_without_intent)

    def _display_name(self, stem_without_intent: str) -> str:
        # keep it readable for user-facing text
        return (
            stem_without_intent.replace("(hayfever)", " (hay fever)")
            .replace("(", " (")
            .replace(")", ")")
        )

    def _load(self) -> None:
        for f in sorted(self.path.glob("intent*.json")):
            with f.open("r", encoding="utf-8") as fp:
                data = json.load(fp)

            stem = f.stem
            disease_part = stem[len("intent") :] if stem.startswith("intent") else stem
            key = self._file_key(disease_part)

            intents: List[IntentItem] = []
            for it in data.get("intents", []):
                intents.append(
                    IntentItem(
                        tag=str(it.get("tag", "")).strip().lower(),
                        patterns=[str(x) for x in it.get("patterns", [])],
                        responses=[str(x) for x in it.get("responses", [])],
                    )
                )

            self.files[key] = IntentFile(
                disease_key=key,
                filename=f.name,
                display_name=disease_part,
                intents=intents,
            )

    def get_file(self, diseases: Sequence[str]) -> Optional[IntentFile]:
        for d in diseases:
            key = norm(d)
            if key in self.files:
                return self.files[key]
        return None


def detect_disease_from_text(text: str, store) -> Optional[str]:
    t = text.lower()

    ABBR = {
        "uti": "urinarytractinfection",
        "flu": "flu",
        "urinary tract infection": "urinary tract infection",
        "urinary": "urinary tract infection",
    }

    for abbr, full in ABBR.items():
        if abbr in t:
            return full

    t_clean = re.sub(r"[^a-z0-9 ]+", " ", t)

    for key in store.files.keys():
        words = re.findall(r"[a-z]+", key)

        # strong match (first 2 words)
        if all(w in t_clean for w in words[:2]):
            return key

        # fallback match
        if any(w in t_clean for w in words):
            return key

    return None


# =========================
# intent type detection
# =========================
def is_symptom_style(text: str) -> bool:
    t = text.lower()
    words = t.split()

    question_words = ["what", "why", "how", "when", "which"]

    # If no question words → likely symptoms input
    if not any(q in words for q in question_words):
        return True

    return False


def detect_intent_type(text: str) -> str:
    """Choose the most likely tag from the user's question.

    This is keyword-based but careful:
    - checks stronger patterns first
    - uses scores instead of one brittle if-chain
    - defaults to symptom for complaint-style user input
    """
    if is_symptom_style(text):
        return "symptom"

    t = norm_keep_spaces(text)

    scores = {tag: 0 for tag in TAG_ORDER}

    # Strong phrase matches first
    for tag, phrases in TAG_KEYWORDS.items():
        for phrase in phrases:
            if phrase in t:
                scores[tag] += 3 if len(phrase.split()) > 1 else 2

    # Extra clues
    if any(
        w in t.split()
        for w in [
            "fever",
            "pain",
            "rash",
            "itching",
            "cough",
            "sneezing",
            "nausea",
            "vomiting",
            "diarrhea",
            "headache",
        ]
    ):
        scores["symptom"] += 1

    if any(
        w in t.split()
        for w in [
            "eat",
            "food",
            "diet",
            "banana",
            "rice",
            "milk",
            "spicy",
            "oily",
            "avoid",
        ]
    ):
        scores["food"] += 1

    if any(
        w in t.split()
        for w in ["tablet", "pill", "dose", "dosage", "medicine", "medication", "drug"]
    ):
        scores["medicine"] += 1

    if any(w in t.split() for w in ["after", "recover", "recovery", "better"]):
        scores["post-recovery"] += 1

    # Common question forms
    if t.startswith("what is") or t.startswith("define"):
        scores["definition"] += 2

    # Choose highest score; fall back to symptom if all are zero
    best_tag = max(scores.items(), key=lambda x: (x[1], -TAG_ORDER.index(x[0])))[0]
    if scores[best_tag] == 0:
        return "symptom"
    return best_tag


# =========================
# TF-IDF matcher
# =========================


class IntentMatcher:
    def __init__(self, intent_file: IntentFile):
        self.intent_file = intent_file

    def match(self, query: str, intent_type: str) -> IntentItem:
        # Filter by the predicted tag first.
        filtered = [it for it in self.intent_file.intents if it.tag == intent_type]

        # If no exact tag match exists, use the full file.
        if not filtered:
            filtered = self.intent_file.intents[:]

        # Build one pattern index over the filtered intents.
        patterns: List[str] = []
        mapping: List[int] = []
        for i, intent in enumerate(filtered):
            for p in intent.patterns:
                patterns.append(norm_keep_spaces(p))
                mapping.append(i)

        # Safety fallback if the file has no patterns.
        if not patterns:
            return random.choice(filtered)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        X = vectorizer.fit_transform(patterns)
        q = vectorizer.transform([norm_keep_spaces(query)])
        sims = cosine_similarity(q, X)[0]

        best_pattern_index = int(sims.argmax())
        best_intent_index = mapping[best_pattern_index]
        return filtered[best_intent_index]


# =========================
# follow-up generator
# =========================


def generate_followups(
    intent_file: IntentFile,
    current_tag: str,
    asked_tags: Optional[Sequence[str]] = None,
) -> List[str]:
    asked = set(asked_tags or [])

    # Prefer tag-specific next questions.
    base = FOLLOWUP_QUESTION_TEMPLATES.get(current_tag, [])[:]

    # Add disease-specific questions from the file itself if possible.
    disease_name = intent_file.display_name.replace("(hayfever)", "hay fever")
    disease_name = disease_name.replace("  ", " ").strip()

    extras = [
        f"What is {disease_name}?",
        f"What causes {disease_name}?",
        f"How can I prevent {disease_name}?",
        f"What should I eat if I have {disease_name}?",
        f"Which medicine is commonly used for {disease_name}?",
        f"What should I do after recovery from {disease_name}?",
    ]

    out: List[str] = []
    for q in base + extras:
        tag = detect_intent_type(q)
        if tag not in asked and q not in out:
            out.append(q)
        if len(out) >= 4:
            break
    return out


# =========================
# chatbot
# =========================
def get_intent_by_tag(file, tag):
    for intent in file.intents:
        if intent.tag == tag:
            return intent
    return None


class MedicalChatbot:
    def __init__(self):
        self.store = IntentStore("intents")
        self.current_file: Optional[IntentFile] = None
        self.last_tag: Optional[str] = None

    def _pick_file(self, diseases: List[str]) -> Optional[IntentFile]:
        file = self.store.get_file(diseases)
        if file:
            return file
        return self.current_file

    def reply(self, text: str, k: int = 4) -> Dict[str, Any]:
        vec, debug = statement_to_bn_input(text)
        raw_topk = find_top_k(vec, k)
        diseases = clean_topk(raw_topk)

        # Disease can stay the same across follow-up questions.
        explicit = detect_disease_from_text(text, self.store)

        if explicit and explicit in self.store.files:
            file = self.store.files[explicit]
        else:
            file = self._pick_file(diseases)
        if not file:
            return {
                "reply": "No disease matched.",
                "disease": None,
                "tag": None,
                "followups": [],
                "diseases": diseases,
                "debug_bn": debug,
            }

        intent_type = detect_intent_type(text)
        intent = get_intent_by_tag(file, intent_type)

        if intent is None:
            intent = IntentMatcher(file).match(text, intent_type)
        response = (
            random.choice(intent.responses)
            if intent.responses
            else "No response found."
        )

        self.current_file = file
        self.last_tag = intent.tag

        followups = generate_followups(file, intent.tag)

        return {
            "reply": response,
            "disease": file.display_name,
            "selected_file": file.filename,
            "tag": intent.tag,
            "followups": followups,
            "diseases": diseases,
            "debug_bn": debug,
            "topk_raw": raw_topk,
            "intent_type": intent_type,
        }

    def reset_context(self) -> None:
        self.current_file = None
        self.last_tag = None


# =========================
# examples
# =========================

EXAMPLES = [
    "fever headache nausea",
    "What is flu?",
    "Why do I get flu?",
    "What are the symptoms of flu?",
    "How can I prevent flu?",
    "What food should I eat in flu?",
    "What medicine is used for flu?",
    "What should I do after flu recovery?",
    "I have sneezing blocked nose watery eyes",
    "How to prevent seasonal allergies?",
    "What medicine for urinary tract infection?",
    "What should I eat if I have UTI?",
]


def demo() -> None:
    bot = MedicalChatbot()
    for i, text in enumerate(EXAMPLES, 1):
        print("=" * 80)
        print(f"Example {i}")
        print("Input:", text)
        out = bot.reply(text)
        print("Detected diseases:", out["diseases"])
        print("Selected disease:", out["disease"])
        print("Selected file:", out["selected_file"])
        print("Predicted tag:", out["tag"])
        print("Intent type:", out["intent_type"])
        print("Reply:", out["reply"])
        print("Follow-up questions:")
        for q in out["followups"]:
            print("-", q)


def chat_loop() -> None:
    bot = MedicalChatbot()
    print("Chat ready. Type 'exit' to quit, or 'reset' to clear disease context.")
    while True:
        text = input("You: ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        if text.lower() == "reset":
            bot.reset_context()
            print("Bot: context cleared.")
            continue
        out = bot.reply(text)
        print("Bot:", out["reply"])
        if out["followups"]:
            print("Follow-ups:")
            for q in out["followups"]:
                print("-", q)


if __name__ == "__main__":
    demo()
    # chat_loop()


# """chat.py
# Pipeline:
# 1) User text -> BN -> top_k diseases
# 2) Map disease -> intent<disease>.json
# 3) TF-IDF match user question with intent patterns
# 4) Return best intent response

# Reliable, deterministic, fast.

# Install:
#     pip install scikit-learn
# """

# from __future__ import annotations

# import json
# import random
# import re
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Sequence

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# from inpToBn import statement_to_bn_input
# from decision import find_top_k


# # ---------------------
# # helpers
# # ---------------------

# def norm(s: str) -> str:
#     return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


# def clean_topk(top_k: Sequence[Any]) -> List[str]:
#     res = []
#     for x in top_k:
#         if isinstance(x, str):
#             res.append(x)
#         elif isinstance(x, (list, tuple)) and x:
#             res.append(str(x[0]))
#         else:
#             res.append(str(x))
#     return res


# # ---------------------
# # data models
# # ---------------------

# @dataclass
# class IntentItem:
#     tag: str
#     patterns: List[str]
#     responses: List[str]


# @dataclass
# class IntentFile:
#     disease: str
#     filename: str
#     intents: List[IntentItem]


# # ---------------------
# # intent loader
# # ---------------------

# class IntentStore:
#     def __init__(self, path: str = "intents"):
#         self.path = Path(path)
#         self.files: Dict[str, IntentFile] = {}
#         self._load()

#     def _key(self, name: str) -> str:
#         return re.sub(r"[^a-z0-9()]+", "", name.lower())

#     def _load(self):
#         for f in self.path.glob("intent*.json"):
#             with open(f, encoding="utf-8") as fp:
#                 data = json.load(fp)

#             disease = f.stem.replace("intent", "")
#             key = self._key(disease)

#             intents = []
#             for it in data.get("intents", []):
#                 intents.append(IntentItem(
#                     tag=it.get("tag", ""),
#                     patterns=it.get("patterns", []),
#                     responses=it.get("responses", [])
#                 ))

#             self.files[key] = IntentFile(key, f.name, intents)

#     def get_file(self, diseases: List[str]) -> Optional[IntentFile]:
#         for d in diseases:
#             k = self._key(d)
#             if k in self.files:
#                 return self.files[k]
#         return None


# # ---------------------
# # TF-IDF matcher
# # ---------------------

# class IntentMatcher:
#     def __init__(self, intent_file: IntentFile):
#         self.intent_file = intent_file
#         self.patterns = []
#         self.map_idx = []

#         for i, intent in enumerate(intent_file.intents):
#             for p in intent.patterns:
#                 self.patterns.append(norm(p))
#                 self.map_idx.append(i)

#         self.vectorizer = TfidfVectorizer()
#         self.X = self.vectorizer.fit_transform(self.patterns)

#     def match(self, query: str) -> IntentItem:
#         q = self.vectorizer.transform([norm(query)])
#         sims = cosine_similarity(q, self.X)[0]

#         best_idx = sims.argmax()
#         intent_idx = self.map_idx[best_idx]

#         return self.intent_file.intents[intent_idx]


# # ---------------------
# # chatbot
# # ---------------------

# class MedicalChatbot:
#     def __init__(self):
#         self.store = IntentStore("intents")

#     def reply(self, text: str, k: int = 4) -> Dict:
#         vec, debug = statement_to_bn_input(text)
#         top_k_raw = find_top_k(vec, k)
#         diseases = clean_topk(top_k_raw)

#         file = self.store.get_file(diseases)
#         if not file:
#             return {"reply": "Could not match disease."}

#         matcher = IntentMatcher(file)
#         intent = matcher.match(text)

#         response = random.choice(intent.responses) if intent.responses else "No response found."

#         return {
#             "symptoms": diseases,
#             "file": file.filename,
#             "tag": intent.tag,
#             "reply": response
#         }


# # ---------------------
# # demo
# # ---------------------

# EXAMPLES = [
#     "fever headache nausea",
#     "sneezing blocked nose watery eyes",
#     "burning urination frequent urination"
# ]


# def demo():
#     bot = MedicalChatbot()
#     for i, t in enumerate(EXAMPLES, 1):
#         print("="*60)
#         print("Input:", t)
#         out = bot.reply(t)
#         print("Detected:", out["symptoms"])
#         print("File:", out["file"])
#         print("Tag:", out["tag"])
#         print("Reply:", out["reply"])


# if __name__ == "__main__":
#     demo()
