from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import streamlit as st
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
    "definition": ["what is", "define", "meaning", "about", "explain", "description", "details", "is it"],
    "cause": ["why", "cause", "causes", "reason", "reason for", "trigger", "because of", "due to"],
    "symptom": ["symptom", "symptoms", "feel", "feeling", "having", "experience", "experiencing", "what are the signs", "signs"],
    "prevention": ["prevent", "prevention", "avoid", "protect", "stop", "reduce chance", "how can i avoid"],
    "food": ["food", "eat", "eating", "diet", "what to eat", "what should i eat", "can i eat", "avoid food", "nutrition"],
    "medicine": ["medicine", "medication", "drug", "tablet", "pill", "treatment", "treat", "cure", "prescribe", "dose", "dosage"],
    "post-recovery": ["after recovery", "recover", "recovery", "after getting better", "after flu", "after treatment", "post recovery", "what next"],
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
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def norm_keep_spaces(s: str) -> str:
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
        return (
            stem_without_intent.replace("(hayfever)", " (hay fever)")
            .replace("_", " ")
            .replace("-", " ")
            .strip()
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
                display_name=self._display_name(disease_part),
                intents=intents,
            )

    def get_file(self, diseases: Sequence[str]) -> Optional[IntentFile]:
        for d in diseases:
            key = norm(d)
            if key in self.files:
                return self.files[key]
        return None


def detect_disease_from_text(text: str, store: IntentStore) -> Optional[str]:
    t = text.lower()

    abbr = {
        "uti": "urinarytractinfection",
        "flu": "flu",
        "urinary tract infection": "urinarytractinfection",
        "urinary": "urinarytractinfection",
    }

    for abbr_key, full in abbr.items():
        if abbr_key in t:
            if full in store.files:
                return full

    t_clean = re.sub(r"[^a-z0-9 ]+", " ", t)

    for key in store.files.keys():
        words = re.findall(r"[a-z]+", key)
        if words and all(w in t_clean for w in words[:2]):
            return key
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
    return not any(q in words for q in question_words)


def detect_intent_type(text: str) -> str:
    if is_symptom_style(text):
        return "symptom"

    t = norm_keep_spaces(text)
    scores = {tag: 0 for tag in TAG_ORDER}

    for tag, phrases in TAG_KEYWORDS.items():
        for phrase in phrases:
            if phrase in t:
                scores[tag] += 3 if len(phrase.split()) > 1 else 2

    tokens = t.split()
    if any(w in tokens for w in ["fever", "pain", "rash", "itching", "cough", "sneezing", "nausea", "vomiting", "diarrhea", "headache"]):
        scores["symptom"] += 1
    if any(w in tokens for w in ["eat", "food", "diet", "banana", "rice", "milk", "spicy", "oily", "avoid"]):
        scores["food"] += 1
    if any(w in tokens for w in ["tablet", "pill", "dose", "dosage", "medicine", "medication", "drug"]):
        scores["medicine"] += 1
    if any(w in tokens for w in ["after", "recover", "recovery", "better"]):
        scores["post-recovery"] += 1

    if t.startswith("what is") or t.startswith("define"):
        scores["definition"] += 2

    best_tag = max(scores.items(), key=lambda x: (x[1], -TAG_ORDER.index(x[0])))[0]
    return best_tag if scores[best_tag] > 0 else "symptom"


# =========================
# TF-IDF matcher
# =========================

class IntentMatcher:
    def __init__(self, intent_file: IntentFile):
        self.intent_file = intent_file

    def match(self, query: str, intent_type: str) -> IntentItem:
        filtered = [it for it in self.intent_file.intents if it.tag == intent_type]
        if not filtered:
            filtered = self.intent_file.intents[:]

        patterns: List[str] = []
        mapping: List[int] = []
        for i, intent in enumerate(filtered):
            for p in intent.patterns:
                patterns.append(norm_keep_spaces(p))
                mapping.append(i)

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

def generate_followups(intent_file: IntentFile, current_tag: str, asked_tags: Optional[Sequence[str]] = None) -> List[str]:
    asked = set(asked_tags or [])
    base = FOLLOWUP_QUESTION_TEMPLATES.get(current_tag, [])[:]

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

def get_intent_by_tag(file: IntentFile, tag: str) -> Optional[IntentItem]:
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
        return file if file else self.current_file

    def reply(self, text: str, k: int = 4) -> Dict[str, Any]:
        vec, debug = statement_to_bn_input(text)
        raw_topk = find_top_k(vec, k)
        diseases = clean_topk(raw_topk)

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

        response = random.choice(intent.responses) if intent.responses else "No response found."

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
# Streamlit UI helpers
# =========================

APP_TITLE = "MediWise"
APP_SUBTITLE = "medical chatbot"


def init_state() -> None:
    if "bot" not in st.session_state:
        st.session_state.bot = MedicalChatbot()
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi, I am MediWise. Describe your symptoms, ask about a disease, or tap a follow-up question.",
                "followups": [],
                "meta": {},
            }
        ]
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "asked_history" not in st.session_state:
        st.session_state.asked_history = []


def add_message(role: str, content: str, followups: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None) -> None:
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "followups": followups or [],
            "meta": meta or {},
        }
    )


def stream_text(text: str, speed: float = 0.012) -> None:
    placeholder = st.empty()
    words = text.split()
    built = []
    for w in words:
        built.append(w)
        placeholder.markdown(" ".join(built) + "▌")
        time.sleep(speed)
    placeholder.markdown(text)


def process_prompt(prompt: str) -> None:
    bot: MedicalChatbot = st.session_state.bot
    result = bot.reply(prompt)

    st.session_state.asked_history.append(prompt)
    add_message("assistant", result["reply"], followups=result.get("followups", []), meta={
        "disease": result.get("disease"),
        "tag": result.get("tag"),
        "intent_type": result.get("intent_type"),
    })


def followup_button_label(text: str) -> str:
    return text if len(text) <= 44 else text[:41] + "..."


# =========================
# page
# =========================

st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="centered")
init_state()

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.1rem; padding-bottom: 7rem; max-width: 860px; }
    .hero {
        padding: 1rem 1rem 0.25rem 1rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(14,165,233,0.10));
        border: 1px solid rgba(148,163,184,0.25);
        margin-bottom: 1rem;
    }
    .hero h1 { font-size: 2rem; margin-bottom: 0.2rem; }
    .hero p { margin-top: 0; opacity: 0.75; }
    div[data-testid="stChatMessage"] {
        border-radius: 20px;
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
    }
    .followup-row {
        margin-top: -0.25rem;
        margin-bottom: 0.8rem;
        padding-left: 0.25rem;
    }
    .followup-note {
        font-size: 0.85rem;
        opacity: 0.72;
        margin: 0.25rem 0 0.55rem 0;
    }
    .footer-note {
        font-size: 0.8rem;
        opacity: 0.65;
        text-align: center;
        margin-top: 0.75rem;
    }
    .stButton > button {
        border-radius: 999px !important;
        border: 1px solid rgba(148,163,184,0.35) !important;
        padding: 0.35rem 0.75rem !important;
        background: rgba(255,255,255,0.04) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <h1>🩺 {APP_TITLE}</h1>
        <p>{APP_SUBTITLE} — streaming replies, clickable follow-ups, and a chat-first layout.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    if st.button("Reset chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi, I am MediWise. Describe your symptoms, ask about a disease, or tap a follow-up question.",
                "followups": [],
                "meta": {},
            }
        ]
        st.session_state.bot.reset_context()
        st.session_state.asked_history = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.caption("Tip: ask with symptoms like: fever headache nausea")
    st.caption("Follow-up chips appear under each assistant reply.")


# render chat history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            if idx == 0:
                stream_text(msg["content"], speed=0.0)
            else:
                st.markdown(msg["content"])
        else:
            st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("followups"):
            st.markdown('<div class="followup-note">Follow-up questions</div>', unsafe_allow_html=True)
            cols = st.columns(min(2, len(msg["followups"])))
            for i, q in enumerate(msg["followups"]):
                with cols[i % len(cols)]:
                    if st.button(followup_button_label(q), key=f"fu_{idx}_{i}_{hash(q)}", use_container_width=True):
                        st.session_state.pending_prompt = q
                        st.rerun()


# handle queued click or user input
prompt = st.chat_input("Type symptoms or a disease question...")
active_prompt = st.session_state.pending_prompt or prompt

if active_prompt:
    st.session_state.pending_prompt = None
    add_message("user", active_prompt)

    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        result = st.session_state.bot.reply(active_prompt)
        reply = result["reply"]
        typed = []
        for token in reply.split():
            typed.append(token)
            placeholder.markdown(" ".join(typed) + "▌")
            time.sleep(0.015)
        placeholder.markdown(reply)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
            "followups": result.get("followups", []),
            "meta": {
                "disease": result.get("disease"),
                "tag": result.get("tag"),
                "intent_type": result.get("intent_type"),
                "selected_file": result.get("selected_file"),
            },
        }
    )
    st.rerun()


st.markdown(
    "<div class='footer-note'>This UI is for chatbot presentation only and is not a medical diagnosis tool.</div>",
    unsafe_allow_html=True,
)
