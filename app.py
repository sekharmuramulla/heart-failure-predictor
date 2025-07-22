
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model(1).pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = [float(request.form[key]) for key in request.form]
        prediction = model.predict([np.array(input_data)])
        result = "Patient is at risk" if prediction[0] == 1 else "Patient is not at risk"
        return render_template("index.html", prediction_text=result)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
