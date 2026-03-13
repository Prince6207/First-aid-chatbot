from flask import Flask,render_template,request
import numpy as np
import pandas as pd
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST':
        name=request.form['username']
        return f"Hello {name}"
    return render_template('name.html')
if __name__=='__main__':
    app.run(debug=True)