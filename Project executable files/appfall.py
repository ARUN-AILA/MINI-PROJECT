import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template
import sklearn

app = Flask(__name__,static_folder='static',template_folder='templates')
model = pickle.load(open('rainfall.pkl1', 'rb'))
scale = pickle.load(open('scale.pkl1', 'rb'))

@app.route("/")
def home():
    return render_template('home (1).html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    # Fetch form data
    input_feature = [request.form.get(key) for key in request.form.keys() if key != 'date']
    
    # Print for debugging
    print(f"Input features: {input_feature}")
    

    features_values = [np.array(input_feature)]
    
    names = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall','Evaporation','Sunshine', 'WindGustDir',
             'WindGustSpeed','WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
             'Pressure9am', 'Pressure3pm', 'Cloud9am','Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'Date_month', 'Date_day']
    
    data = pd.DataFrame(features_values, columns=names)
    print(data.columns)
    
    # Print for debugging
    print(f"Data before scaling: \n{data}")
    data = scale.transform(data)  # Use transform instead of fit_transform
    data = pd.DataFrame(data, columns=names)
    
    # Print for debugging
    print(f"Data after scaling: \n{data}")
    
    # Predictions using the loaded model files
    prediction = model.predict(data)
    pred_prob = model.predict_proba(data)
    
    print(f"Prediction: {prediction}")
    if prediction[0] == "NO":
        return render_template("rainy.html")
    else:
        return render_template("sunny.html")

if __name__ == "__main__":
    app.run(debug=True)
