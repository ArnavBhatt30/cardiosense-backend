from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model    = joblib.load('model.pkl')
scaler   = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'CardioSense API running'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        vals = [float(data[f]) for f in features]
        arr  = np.array(vals).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        prob = float(model.predict_proba(arr_scaled)[0][1])
        pred = int(prob >= 0.5)
        return jsonify({
            'prediction':  pred,
            'probability': round(prob, 4),
            'risk': 'high' if prob >= 0.6 else 'mid' if prob >= 0.4 else 'low'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
