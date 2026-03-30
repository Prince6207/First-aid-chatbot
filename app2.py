import streamlit as st
import pickle
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import json
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Input
import nltk
from nltk.stem import PorterStemmer
import numpy as np
if 'predicted_disease' not in st.session_state:
    st.session_state.predicted_disease = None
if 'disease_id' not in st.session_state:
    st.session_state.disease_id = None
stemmer=PorterStemmer()
model=None
with open('model.pkl','rb') as f:
    model=pickle.load(f)
symptoms=['depression', 'shortnessofbreath', 'depressiveorpsychoticsymptoms', 'sharpchestpain', 'dizziness', 'abnormalinvoluntarymovements', 'sorethroat', 'cough', 'nasalcongestion', 'throatswelling', 'diminishedhearing', 'lumpinthroat', 'throatfeelstight', 'skinswelling', 'retentionofurine', 'legpain', 'suprapubicpain', 'lackofgrowth', 'elbowweakness', 'whitedischargefromeye', 'abusingalcohol', 'fainting', 'drugabuse', 'sharpabdominalpain', 'vomiting', 'headache', 'nausea', 'diarrhea', 'vaginalitching', 'painfulurination', 'frequenturination', 'lowerabdominalpain', 'vaginaldischarge', 'bloodinurine', 'wristpain', 'handorfingerswelling', 'armpain', 'lipswelling', 'abnormalappearingskin', 'skinlesion', 'acneorpimples', 'mouthulcer', 'diminishedvision', 'painineye', 'irregularappearingscalp', 'backpain', 'neckpain', 'lowbackpain', 'pelvicpain', 'vomitingblood', 'wheezing', 'peripheraledema', 'earpain', 'footortoepain', 'skinmoles', 'kneelumpormass', 'vaginalpain', 'weakness', 'ringinginear', 'pluggedfeelinginear', 'frontalheadache', 'fluidinear', 'spotsorcloudsinvision', 'eyeredness', 'lacrimation', 'itchinessofeye', 'blindness', 'lossofsensation', 'slurringwords', 'symptomsoftheface', 'disturbanceofmemory', 'sidepain', 'fever', 'acheallover', 'changesinstoolappearance', 'chills', 'fatigue', 'melena', 'coryza', 'allergicreaction', 'sleepiness', 'abnormalbreathingsounds', 'pullingatears', 'rednessinear', 'fluidretention', 'flu-likesyndrome', 'sinuscongestion', 'musclecramps,contractures,orspasms', 'nosebleed', 'swolleneye', 'itchingofskin', 'skindryness,peeling,scaliness,orroughness', 'skinrash', 'feelinghot', 'swollenorredtonsils', 'lipsore', 'sneezing', 'diaperrash', 'throatredness'] 
infer=VariableElimination(model)
evidence=[]
with st.form(key='symptoms_form'):
    evidence = []
    for sym in symptoms:
        evidence.append(st.radio(sym + ' :', ['No', 'Yes']))
    submit_button = st.form_submit_button('Submit')
if submit_button:
    evidence2={}
    for i in range(len(symptoms)):
        evidence2[symptoms[i]]=1 if evidence[i]=='Yes' else 0
    pred=infer.query(variables=['diseases'],evidence=evidence2)
    ans={}
    dis=pred.state_names
    dis=dis['diseases']
    for i in range(len(dis)):
        ans[dis[i]]=pred.values[i]
    for i in ans:
        st.write(f'{i} : {ans[i]}')
    predicted_name=None
    maxprob=0
    for i in ans:
        if ans[i]>maxprob:
            maxprob=ans[i]
            predicted_name=i
    st.session_state.predicted_disease = predicted_name
    st.session_state.disease_id = predicted_name.replace(' ', '')
if st.session_state.predicted_disease:
    pred_name = st.session_state.predicted_disease
    i = st.session_state.disease_id
    rawjson=r'C:\Users\princ\OneDrive\Desktop\AI Project\intents\intent'
    rawmodel=r'C:\Users\princ\OneDrive\Desktop\AI Project\models\model'
    with open(rawjson+i+'.json') as f:
        data=json.load(f)
    data=data['intents']
    words=[]
    docs_x=[]
    labels=[]
    docs_y=[]
    for doc in data:
        labels.append(doc['tag'])
        for pattern in doc['patterns']:
            token=nltk.word_tokenize(pattern)
            wrds=[stemmer.stem(w.lower()) for w in token]
            words.extend(wrds)
            docs_x.append(pattern)
            docs_y.append(doc['tag'])
    words=sorted(list(set(words)))
    model=load_model(rawmodel+i+'.keras')
    st.write('Type "quit" to exit')
    with st.form(key='chat_form', clear_on_submit=True):
        ques = st.text_input(f'Ask something about {pred_name}')
        chat_submit = st.form_submit_button('Ask')
    if chat_submit and ques:
        st.write(f"You: {ques}")
        if ques.lower() == 'quit':
            st.session_state.predicted_disease = None
            st.session_state.disease_id = None
            st.rerun()
        else:
            wrds = nltk.word_tokenize(ques)
            inp = [stemmer.stem(w.lower()) for w in wrds]
            input_i = []
            
            for word in words:
                if word in inp:
                    input_i.append(1)
                else:
                    input_i.append(0)
                    
            input_i = np.array([input_i])
            pred_chat = model.predict(input_i)[0]
            idx = np.argmax(pred_chat)
            
            if pred_chat[idx] < 0.2:
                st.write('Bot: I am not sure')
            else:
                result = labels[idx]
                for doc in data:
                    if doc['tag'] == result:
                        st.write(f"Bot: {doc['responses'][0]}") 
                        break