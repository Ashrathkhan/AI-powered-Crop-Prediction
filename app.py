from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load saved files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]

        data = scaler.transform([data])
        prediction = model.predict(data)
        result = le.inverse_transform(prediction)

        return render_template('index.html', prediction_text=f"Recommended Crop: {result[0]}")

    except:
        return render_template('index.html', prediction_text="Error in input")

if __name__ == "__main__":
    app.run(debug=True)