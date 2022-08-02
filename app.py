import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle


app = Flask(__name__)
model = pickle.load(open('KNN_Titanic.pkl','rb'))
dataset = pd.read_csv('train.csv')
X=dataset.iloc[:,[2,4,5,6,7,9]].values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    
    '''
    pclass = int(request.args.get('pclass'))

    age = int(request.args.get('age'))
    sibsp = int(request.args.get('sibsp'))
    parch = int(request.args.get('parch'))
    fare = float(request.args.get('fare'))

    sex = request.args.get('sex')
    if sex=="Male":
      sex = 1
    else:
      sex = 0

    
    
    prediction = model.predict(sc.transform([[pclass, sex, age, sibsp, parch, fare]]))
    
    print("Survived", prediction)
    if prediction==[1]:
        prediction="The Person Survived"
    else:
        prediction="The Person Succumbed"
    print(prediction)
        
    return render_template('index.html', prediction_text='Classification Model has predicted the survival based on various parameters: {}'.format(prediction))
    
  

if __name__=="__main__":
    app.run(debug=True)


