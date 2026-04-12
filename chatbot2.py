from __future__ import annotations

import json
import random
import re
import time
import pickle
import torch
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import streamlit as st

# Local imports from your project
from inpToBn import statement_to_bn_input
from decision import find_top_k

# =========================
# 1. RNN Model & Vocab Classes
# =========================

class Vocab:
    def __init__(self, max_vocab=5000):
        self.word2idx = {"<PAD>": 0, "<OOV>": 1}
        self.idx2word = {0: "<PAD>", 1: "<OOV>"}
        self.max_vocab = max_vocab

    def encode(self, text):
        return [self.word2idx.get(w, 1) for w in text.lower().split()]

# =========================
# 2. Config & Constants
# =========================

TAG_ORDER = ["definition", "cause", "symptom", "prevention", "food", "medicine", "post-recovery"]
ID2LABEL = {i: label for i, label in enumerate(TAG_ORDER)}

# =========================
# 3. Helpers & Data Classes
# =========================

def clean_topk(top_k: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for x in top_k:
        if isinstance(x, str): out.append(x)
        elif isinstance(x, (list, tuple)) and x: out.append(str(x[0]))
    return out

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

class IntentStore:
    def __init__(self, path: str = "intents"):
        self.path = Path(path)
        self.files: Dict[str, IntentFile] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists(): return
        for f in sorted(self.path.glob("intent*.json")):
            with f.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            stem = f.stem
            disease_part = stem[len("intent"):] if stem.startswith("intent") else stem
            key = re.sub(r"[^a-z0-9]+", "", disease_part.lower())
            intents = [IntentItem(tag=str(it.get("tag", "")).strip().lower(),
                                  patterns=[str(x) for x in it.get("patterns", [])],
                                  responses=[str(x) for x in it.get("responses", [])]) 
                       for it in data.get("intents", [])]
            self.files[key] = IntentFile(disease_key=key, filename=f.name, 
                                         display_name=disease_part.replace("_", " ").strip(), 
                                         intents=intents)

    def get_file(self, diseases: Sequence[str]) -> Optional[IntentFile]:
        for d in diseases:
            key = re.sub(r"[^a-z0-9]+", "", d.lower())
            if key in self.files: return self.files[key]
        return None

# =========================
# 4. Core Chatbot Logic
# =========================

@st.cache_resource
def load_medical_model():
    model = torch.jit.load("bilstm_model/model.pt")
    model.eval()
    with open("bilstm_model/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    return model, vocab

class MedicalChatbot:
    def __init__(self):
        self.store = IntentStore("intents")
        self.current_file: Optional[IntentFile] = None
        try:
            self.model, self.vocab = load_medical_model()
            self.model_loaded = True
        except Exception:
            self.model_loaded = False

    def predict_intent_rnn(self, text, max_len=64, oov_threshold=0.25, confidence_threshold=0.6, temperature=2.0):
        if not self.model_loaded: return None
        
        tokens = text.lower().split()
        encoded = []
        oov_count = 0
        for w in tokens:
            if w in self.vocab.word2idx: encoded.append(self.vocab.word2idx[w])
            else:
                encoded.append(1)
                oov_count += 1
        
        if (oov_count / max(len(tokens), 1)) > oov_threshold: return None

        seq = encoded[:max_len] + [0] * (max_len - len(encoded))
        x = torch.tensor([seq]).to(next(self.model.parameters()).device)

        with torch.no_grad():
            logits = self.model(x) / temperature
            probs = torch.softmax(logits, dim=1)[0]
        
        confidence, label_id = torch.max(probs, dim=0)
        return ID2LABEL.get(label_id.item()) if confidence.item() >= confidence_threshold else None

    def reply(self, text: str) -> Dict[str, Any]:
        # 1. Disease Prediction (BN)
        vec, debug = statement_to_bn_input(text)
        diseases = clean_topk(find_top_k(vec, 4))
        
        # 2. Intent Prediction (RNN Only)
        rnn_tag = self.predict_intent_rnn(text)
        
        # Analysis metadata for the toggle button
        analysis_data = {
            "rnn_tag": rnn_tag,
            "bn_diseases": diseases
        }

        # Handle Missing Intent or Context
        file = self.store.get_file(diseases) or self.current_file
        
        if rnn_tag is None or file is None:
            return {
                "reply": "I'm not sure which condition you're referring to. Could you describe your symptoms more?",
                "followups": [],
                "analysis": analysis_data
            }

        # 3. Generate Response
        intent = next((it for it in file.intents if it.tag == rnn_tag), None)
        response = random.choice(intent.responses) if intent else f"I'm not sure about the {rnn_tag} for {file.display_name}."

        self.current_file = file
        return {
            "reply": response,
            "disease": file.display_name,
            "tag": rnn_tag,
            "followups": self.get_followups(file),
            "analysis": analysis_data
        }

    def get_followups(self, file):
        d = file.display_name
        return [f"What is {d}?", f"What are the causes of {d}?", f"Treatment for {d}?"]

# =========================
# 5. Streamlit UI
# =========================

st.set_page_config(page_title="MediWise", page_icon="🩺")

if "bot" not in st.session_state:
    st.session_state.bot = MedicalChatbot()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, I am MediWise. How can I help you?", "followups": [], "analysis": None}]

# Sidebar
with st.sidebar:
    if st.button("Reset Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Hi, I am MediWise.", "followups": [], "analysis": None}]
        st.session_state.bot.current_file = None
        st.rerun()

# Render History
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Unique Analysis Toggle per response
        if msg["role"] == "assistant" and msg.get("analysis"):
            with st.expander("🔍 Analysis"):
                data = msg["analysis"]
                st.write(f"**RNN Tag:** `{data['rnn_tag']}`")
                st.write(f"**Disease By BN:** `{', '.join(data['bn_diseases'])}`")

        # Followups
        if msg["role"] == "assistant" and msg.get("followups"):
            cols = st.columns(len(msg["followups"]))
            for j, q in enumerate(msg["followups"]):
                if cols[j].button(q, key=f"fup_{i}_{j}"):
                    st.session_state.user_input = q
                    st.rerun()

# Handle Input
user_query = st.chat_input("Describe your symptoms...")
if "user_input" in st.session_state:
    user_query = st.session_state.user_input
    del st.session_state.user_input

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        res = st.session_state.bot.reply(user_query)
        
        # Typing Effect
        full_text = res["reply"]
        curr = ""
        for word in full_text.split():
            curr += word + " "
            placeholder.markdown(curr + "▌")
            time.sleep(0.02)
        placeholder.markdown(full_text)

    # Store message with unique analysis data
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_text, 
        "followups": res["followups"],
        "analysis": res["analysis"]
    })
    st.rerun()