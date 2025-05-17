# app.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained pipeline model
model = joblib.load('models/model_pipeline.pkl')

@app.route('/')
def home():
    return "Welcome to the Insurance Premium Prediction API!"

@app.route('/predict_file', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(file)
        
        # Perform prediction
        predictions = model.predict(df)
        df['predicted_charges'] = predictions

        # Convert to list of dicts for JSON response
        result = df.to_dict(orient='records')

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
