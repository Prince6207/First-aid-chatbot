import streamlit as st
import pickle
import json
import nltk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pgmpy.inference import VariableElimination

# --- Initialization ---
if 'predicted_disease' not in st.session_state:
    st.session_state.predicted_disease = None
if 'disease_id' not in st.session_state:
    st.session_state.disease_id = None

# Load the Bayesian Network for Initial Diagnosis
@st.cache_resource
def load_bayesian_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

bayesian_model = load_bayesian_model()
infer = VariableElimination(bayesian_model)

symptoms = ['depression', 'shortnessofbreath', 'depressiveorpsychoticsymptoms', 'sharpchestpain', 'dizziness', 'abnormalinvoluntarymovements', 'sorethroat', 'cough', 'nasalcongestion', 'throatswelling', 'diminishedhearing', 'lumpinthroat', 'throatfeelstight', 'skinswelling', 'retentionofurine', 'legpain', 'suprapubicpain', 'lackofgrowth', 'elbowweakness', 'whitedischargefromeye', 'abusingalcohol', 'fainting', 'drugabuse', 'sharpabdominalpain', 'vomiting', 'headache', 'nausea', 'diarrhea', 'vaginalitching', 'painfulurination', 'frequenturination', 'lowerabdominalpain', 'vaginaldischarge', 'bloodinurine', 'wristpain', 'handorfingerswelling', 'armpain', 'lipswelling', 'abnormalappearingskin', 'skinlesion', 'acneorpimples', 'mouthulcer', 'diminishedvision', 'painineye', 'irregularappearingscalp', 'backpain', 'neckpain', 'lowbackpain', 'pelvicpain', 'vomitingblood', 'wheezing', 'peripheraledema', 'earpain', 'footortoepain', 'skinmoles', 'kneelumpormass', 'vaginalpain', 'weakness', 'ringinginear', 'pluggedfeelinginear', 'frontalheadache', 'fluidinear', 'spotsorcloudsinvision', 'eyeredness', 'lacrimation', 'itchinessofeye', 'blindness', 'lossofsensation', 'slurringwords', 'symptomsoftheface', 'disturbanceofmemory', 'sidepain', 'fever', 'acheallover', 'changesinstoolappearance', 'chills', 'fatigue', 'melena', 'coryza', 'allergicreaction', 'sleepiness', 'abnormalbreathingsounds', 'pullingatears', 'rednessinear', 'fluidretention', 'flu-likesyndrome', 'sinuscongestion', 'musclecramps,contractures,orspasms', 'nosebleed', 'swolleneye', 'itchingofskin', 'skindryness,peeling,scaliness,orroughness', 'skinrash', 'feelinghot', 'swollenorredtonsils', 'lipsore', 'sneezing', 'diaperrash', 'throatredness']

# --- Step 1: Initial Symptom Form ---
st.title("AI Medical Assistant")
with st.form(key='symptoms_form'):
    st.subheader("Select your symptoms:")
    responses = []
    # Using columns to make the form less vertical
    cols = st.columns(3)
    for index, sym in enumerate(symptoms):
        with cols[index % 3]:
            responses.append(st.radio(f"{sym}:", ['No', 'Yes'], key=sym))
    
    submit_button = st.form_submit_button('Diagnose')

if submit_button:
    evidence2 = {symptoms[i]: (1 if responses[i] == 'Yes' else 0) for i in range(len(symptoms))}
    pred = infer.query(variables=['diseases'], evidence=evidence2)
    
    # Get highest probability disease
    dis_states = pred.state_names['diseases']
    probs = pred.values
    idx = np.argmax(probs)
    
    st.session_state.predicted_disease = dis_states[idx]
    st.session_state.disease_id = dis_states[idx].lower().replace(' ', '')
    st.success(f"Preliminary Diagnosis: {st.session_state.predicted_disease} (Confidence: {probs[idx]:.2%})")

# --- Step 2: Specialized Chatbot for Predicted Disease ---
if st.session_state.predicted_disease:
    pred_name = st.session_state.predicted_disease
    d_id = st.session_state.disease_id
    
    # Paths
    json_path = f'C:\\Users\\princ\\OneDrive\\Desktop\\AI Project\\intents\\intent{d_id}.json'
    model_path = f'C:\\Users\\princ\\OneDrive\\Desktop\\AI Project\\models3\\model{d_id}.h5'
    
    try:
        with open(json_path) as f:
            intents_data = json.load(f)['intents']
        
        # Load RNN Model
        chat_model = load_model(model_path)
        
        # Prepare Tokenizer and max_len (Must match Training)
        all_patterns = []
        labels = []
        for intent in intents_data:
            for pattern in intent['patterns']:
                all_patterns.append(pattern.lower())
            labels.append(intent['tag'])
            
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_patterns)
        max_len = max([len(nltk.word_tokenize(p)) for p in all_patterns])

        st.divider()
        st.subheader(f"Chat with our {pred_name} specialist")
        
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input(f"Ask about {pred_name} (symptoms, treatment, etc.):")
            chat_submit = st.form_submit_button('Send')
        
        if chat_submit and user_input:
            if user_input.lower() == 'quit':
                st.session_state.predicted_disease = None
                st.rerun()
            
            # Preprocess input for RNN
            seq = tokenizer.texts_to_sequences([user_input.lower()])
            padded = pad_sequences(seq, maxlen=max_len, padding='post')
            
            # Predict
            pred_probs = chat_model.predict(padded)[0]
            pred_idx = np.argmax(pred_probs)
            
            if pred_probs[pred_idx] < 0.3:
                st.info("Bot: I'm not entirely sure about that. Could you rephrase?")
            else:
                tag = labels[pred_idx]
                for intent in intents_data:
                    if intent['tag'] == tag:
                        st.markdown(f"**You:** {user_input}")
                        st.markdown(f"**Bot:** {intent['responses'][0]}")
                        break
        
        if st.button("Reset Diagnosis"):
            st.session_state.predicted_disease = None
            st.rerun()

    except FileNotFoundError:
        st.error(f"Error: Model or Intent file for '{d_id}' not found.")