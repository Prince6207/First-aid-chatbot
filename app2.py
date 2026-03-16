import streamlit as st
import pickle
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
model=None
with open('model.pkl','rb') as f:
    model=pickle.load(f)
symptoms=['depression', 'shortnessofbreath', 'depressiveorpsychoticsymptoms', 'sharpchestpain', 'dizziness', 'abnormalinvoluntarymovements', 'sorethroat', 'cough', 'nasalcongestion', 'throatswelling', 'diminishedhearing', 'lumpinthroat', 'throatfeelstight', 'skinswelling', 'retentionofurine', 'legpain', 'suprapubicpain', 'lackofgrowth', 'elbowweakness', 'whitedischargefromeye', 'abusingalcohol', 'fainting', 'drugabuse', 'sharpabdominalpain', 'vomiting', 'headache', 'nausea', 'diarrhea', 'vaginalitching', 'painfulurination', 'frequenturination', 'lowerabdominalpain', 'vaginaldischarge', 'bloodinurine', 'wristpain', 'handorfingerswelling', 'armpain', 'lipswelling', 'abnormalappearingskin', 'skinlesion', 'acneorpimples', 'mouthulcer', 'diminishedvision', 'painineye', 'irregularappearingscalp', 'backpain', 'neckpain', 'lowbackpain', 'pelvicpain', 'vomitingblood', 'wheezing', 'peripheraledema', 'earpain', 'footortoepain', 'skinmoles', 'kneelumpormass', 'vaginalpain', 'weakness', 'ringinginear', 'pluggedfeelinginear', 'frontalheadache', 'fluidinear', 'spotsorcloudsinvision', 'eyeredness', 'lacrimation', 'itchinessofeye', 'blindness', 'lossofsensation', 'slurringwords', 'symptomsoftheface', 'disturbanceofmemory', 'sidepain', 'fever', 'acheallover', 'changesinstoolappearance', 'chills', 'fatigue', 'melena', 'coryza', 'allergicreaction', 'sleepiness', 'abnormalbreathingsounds', 'pullingatears', 'rednessinear', 'fluidretention', 'flu-likesyndrome', 'sinuscongestion', 'musclecramps,contractures,orspasms', 'nosebleed', 'swolleneye', 'itchingofskin', 'skindryness,peeling,scaliness,orroughness', 'skinrash', 'feelinghot', 'swollenorredtonsils', 'lipsore', 'sneezing', 'diaperrash', 'throatredness'] 
infer=VariableElimination(model)
evidence=[]
for sym in symptoms:
    evidence.append(st.radio(sym+' :',['No','Yes']))
if st.button('Submit'):
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