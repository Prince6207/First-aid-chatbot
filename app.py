from flask import Flask,render_template,request
import numpy as np

import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
df=pd.read_csv('new_disease_dataset.csv')
df=df.drop(columns=['Unnamed: 0'])
symptoms=df.columns
symptoms=symptoms[1:]
columns={}
for sym in symptoms:
    columns[sym]=sym.replace(' ','')
df=df.rename(columns=columns)
symptoms=[symp for symp in df.columns if symp!='diseases']
edges=[('diseases',sym) for sym in symptoms]
model=DiscreteBayesianNetwork(edges)
from sklearn.model_selection import train_test_split
X=df.drop(columns=['diseases'])
y=df['diseases']
X_train, y_train=df.iloc[:,1:],df.iloc[:,0]
train_data=pd.concat([y_train,X_train],axis=1)
diseases=y_train.unique().tolist()
all_state_names = {'diseases': diseases}
for symptom in symptoms:
    all_state_names[symptom] = [0, 1]

model.fit(data=train_data, estimator=BayesianEstimator, state_names=all_state_names)
infer=VariableElimination(model)
symptoms=df.columns[1:]
    
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST':
        evidence={}
        for i,j in request.form.items():
            evidence[i]=int(j)
        pred=infer.query(variables=['diseases'],evidence=evidence)
        ans={}
        print(pred)
        dis=pred.state_names
        dis=dis['diseases']
        for i in range(len(diseases)):
            ans[dis[i]]=pred.values[i]
        return ans
    return render_template('name.html',symptoms=symptoms)
if __name__=='__main__':
    app.run(debug=True)