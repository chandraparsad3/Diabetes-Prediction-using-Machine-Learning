from flask import Flask
import pickle
from flask import request,jsonify,render_template
from sklearn.preprocessing  import StandardScaler
import numpy as np

app = Flask(__name__)
prediction=pickle.load(open('models/prediciton.pkl','rb'))
scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def hello_world():
    return "<h1>Hello, World!</h1>"

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_data():
    if request.method =='POST':
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['bloodPressure'])
        skin_thickness = float(request.form['skinThickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_function = float(request.form['diabetesPedigreeFunction'])
        age = float(request.form['age'])
        
        features=scaler.transform([[pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree_function,age]])
        result=prediction.predict(features)
        
        if(result[0]>0):
            result="The person has diabetes"
        else :
            result="The person doesnot has diabetes"
         

        return render_template('predict.html',result=result)
    else:
        return render_template('predict.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
