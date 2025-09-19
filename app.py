# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        bp = float(request.form["bp"])
        cholesterol = float(request.form["cholesterol"])
        features = np.array([[age, bp, cholesterol]])
        prediction = model.predict(features)[0]
        return render_template("index.html", prediction=prediction)
    except:
        return "Error in input data!"

if __name__ == "__main__":
    app.run(debug=True,port=5001)
