import os
import logging
import warnings

# ==========================================
# 0. SILENCE NOISY LOGS
# ==========================================
# This stops sentence_transformers and httpx from spamming your terminal
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

import streamlit as st
import json
import random
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Import your custom modules
from inpToBn import statement_to_bn_input
from decision import find_top_k

# ==========================================
# 1. INITIALIZATION & HARDWARE-CACHED LOADING
# ==========================================
st.set_page_config(page_title="First-Aid AI Assistant", page_icon="🩺", layout="wide")

@st.cache_resource
def load_intent_classifier():
    """Loads and caches the SVM model, label encoder, and embedding text engine."""
    with open("svm_intent_model.pkl", "rb") as f:
        svm_model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return svm_model, label_encoder, embedder

try:
    svm_model, label_encoder, embedder = load_intent_classifier()
except FileNotFoundError:
    st.error("Target classification models ('svm_intent_model.pkl' or 'label_encoder.pkl') missing. Please check model storage pathways.")
    st.stop()

# Track active diagnoses and chat logs across app refreshes
if "detected_disease" not in st.session_state:
    st.session_state.detected_disease = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "disease_knowledge" not in st.session_state:
    st.session_state.disease_knowledge = None

# Configurable decision boundary for tracking out-of-scope/unmatched intents
CONFIDENCE_THRESHOLD = 0.55
FALLBACK_RESPONSE = "I am a First-Aid assistant focused on core health details. I don't have accurate data regarding that question. Please ask about the definition, cause, symptoms, prevention, food, medicine, or post-recovery of this condition."

# ==========================================
# 2. FILE SELECTION HELPER
# ==========================================
def load_disease_json(disease_name):
    """
    Looks for the matching intent file inside ../intents/ using common naming patterns.
    """
    clean_name = disease_name.strip()
    base_dir = os.path.join("..", "intents")   # go up one level, then into intents/

    # Generate potential file naming variants
    file_variants = [
        f"{clean_name}intent.json",
        f"{clean_name.lower()}intent.json",
        f"intent_{clean_name}.json",
        f"{clean_name}_intent.json"
    ]

    for variant in file_variants:
        filepath = os.path.join(base_dir, variant)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
    return None

# ==========================================
# 3. INTERACTION INTERFACE
# ==========================================
st.title("🩺 Smart First-Aid Assistant & Diagnostic Engine")
st.write("Enter your symptoms naturally. The system evaluates the underlying probabilities and opens a specialized context window to answer questions.")

# --- STEP 1: NARRATIVE SYMPTOM ENTRY ---
st.subheader("Step 1: Symptom Analysis")
user_symptoms_input = st.text_area(
    "Describe what you are experiencing:", 
    placeholder="e.g., I have been experiencing a sharp chest pain along with shortness of breath, but I don't have any cough.",
    key="symptom_text"
)

if st.button("Run Diagnostic Evaluation", type="primary"):
    if user_symptoms_input.strip() == "":
        st.warning("Please type a valid sentence before executing analysis.")
    else:
        with st.spinner("Processing medical variables and executing inference..."):
            # Execute semantic extraction from inpToBn layer
            evidence_vector, extracted_list = statement_to_bn_input(user_symptoms_input)
            
            # Run inference using pgmpy model execution pipeline in decision.py
            top_diagnoses = find_top_k(evidence_vector, k=3, model_path="model.pkl")
            
        if not top_diagnoses or top_diagnoses[0][0] == "No symptoms detected":
            st.error("No core symptoms could be mapped confidently to the Bayesian Network schema. Please refine your entry.")
        else:
            st.success(f"Symptom extraction complete! Parsed tags: {', '.join(extracted_list)}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Top Bayesian Network Predictions")
                for disease, prob in top_diagnoses:
                    st.metric(label=disease, value=f"{prob * 100:.2f}%")
            
            # Select the most likely prediction
            primary_disease = top_diagnoses[0][0]
            st.session_state.detected_disease = primary_disease
            
            # Clear chat logs when switching to a brand new condition
            st.session_state.chat_history = [] 
            
            # Load targeted knowledge graph
            knowledge_base = load_disease_json(primary_disease)
            if knowledge_base:
                st.session_state.disease_knowledge = knowledge_base
                st.toast(f"Loaded context file for {primary_disease}", icon="📂")
            else:
                st.session_state.disease_knowledge = None
                st.error(f"Could not locate a matching intent file for '{primary_disease}'. Verify JSON file naming matches context patterns.")

# --- STEP 2: DYNAMIC CONTEXTUAL KNOWLEDGE CHAT ---
if st.session_state.detected_disease and st.session_state.disease_knowledge:
    st.write("---")
    st.subheader(f"Step 2: Consultative Assistant — Context: **{st.session_state.detected_disease}**")
    st.info(f"The assistant is now configured to answer direct questions about **{st.session_state.detected_disease}** based on definition, cause, symptom, prevention, food, medicine, and post-recovery profiles.")

    # Render persistent conversation window
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Collect conversational input
    if user_query := st.chat_input(f"Ask about {st.session_state.detected_disease}:"):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        # 1. Transform raw text string into dense sentence vector
        query_vector = embedder.encode([user_query])
        
        # 2. Extract classification probabilities via trained SVM
        probabilities = svm_model.predict_proba(query_vector)[0]
        max_idx = np.argmax(probabilities)
        max_confidence = probabilities[max_idx]
        
        # 3. Handle intentional classification boundaries or fallbacks
        if max_confidence < CONFIDENCE_THRESHOLD:
            predicted_tag = "None"
        else:
            predicted_tag = label_encoder.inverse_transform([max_idx])[0]

        # 4. Extract targeted match string inside active JSON dictionary
        final_reply = FALLBACK_RESPONSE
        
        if predicted_tag != "None":
            intents_list = st.session_state.disease_knowledge.get("intents", [])
            for intent in intents_list:
                if intent.get("tag") == predicted_tag:
                    responses = intent.get("responses", [])
                    if responses:
                        final_reply = random.choice(responses)
                    break
        
        st.session_state.chat_history.append({"role": "assistant", "content": final_reply})
        with st.chat_message("assistant"):
            st.write(final_reply)
            st.caption(f"Classification Intent: **{predicted_tag}** | Internal Score: {max_confidence:.2f}")