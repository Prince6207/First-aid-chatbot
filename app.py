from flask import Flask,render_template,request
import numpy as np
import pickle
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
with open('model.pkl','rb') as f:
    model=pickle.load(f)
infer=VariableElimination(model)
symptoms=['depression', 'shortnessofbreath', 'depressiveorpsychoticsymptoms', 'sharpchestpain', 'dizziness', 'abnormalinvoluntarymovements', 'sorethroat', 'cough', 'nasalcongestion', 'throatswelling', 'diminishedhearing', 'lumpinthroat', 'throatfeelstight', 'skinswelling', 'retentionofurine', 'legpain', 'suprapubicpain', 'lackofgrowth', 'elbowweakness', 'whitedischargefromeye', 'abusingalcohol', 'fainting', 'drugabuse', 'sharpabdominalpain', 'vomiting', 'headache', 'nausea', 'diarrhea', 'vaginalitching', 'painfulurination', 'frequenturination', 'lowerabdominalpain', 'vaginaldischarge', 'bloodinurine', 'wristpain', 'handorfingerswelling', 'armpain', 'lipswelling', 'abnormalappearingskin', 'skinlesion', 'acneorpimples', 'mouthulcer', 'diminishedvision', 'painineye', 'irregularappearingscalp', 'backpain', 'neckpain', 'lowbackpain', 'pelvicpain', 'vomitingblood', 'wheezing', 'peripheraledema', 'earpain', 'footortoepain', 'skinmoles', 'kneelumpormass', 'vaginalpain', 'weakness', 'ringinginear', 'pluggedfeelinginear', 'frontalheadache', 'fluidinear', 'spotsorcloudsinvision', 'eyeredness', 'lacrimation', 'itchinessofeye', 'blindness', 'lossofsensation', 'slurringwords', 'symptomsoftheface', 'disturbanceofmemory', 'sidepain', 'fever', 'acheallover', 'changesinstoolappearance', 'chills', 'fatigue', 'melena', 'coryza', 'allergicreaction', 'sleepiness', 'abnormalbreathingsounds', 'pullingatears', 'rednessinear', 'fluidretention', 'flu-likesyndrome', 'sinuscongestion', 'musclecramps,contractures,orspasms', 'nosebleed', 'swolleneye', 'itchingofskin', 'skindryness,peeling,scaliness,orroughness', 'skinrash', 'feelinghot', 'swollenorredtonsils', 'lipsore', 'sneezing', 'diaperrash', 'throatredness']   
app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def login():
    if request.method=='POST':
        evidence={}
        for i,j in request.form.items():
            evidence[i]=int(j)
        pred=infer.query(variables=['diseases'],evidence=evidence)
        ans={}
        dis=pred.state_names
        dis=dis['diseases']
        for i in range(len(dis)):
            ans[dis[i]]=pred.values[i]
        return ans
    return render_template('name.html',symptoms=symptoms)
if __name__=='__main__':
    app.run(debug=True)