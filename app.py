from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# ✅ Initialize app
app = Flask(__name__)
CORS(app)

# ✅ Load trained model & scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# ✅ Health check route
@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'CardioSense API running'})


# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # ✅ Input features (same as training)
        vals = [
            float(data['age']),
            float(data['gender']),
            float(data['height']),
            float(data['weight']),
            float(data['ap_hi']),
            float(data['ap_lo']),
            float(data['cholesterol']),
            float(data['gluc']),
            float(data['smoke']),
            float(data['alco']),
            float(data['active'])
        ]

        # Convert to array
        arr = np.array(vals).reshape(1, -1)

        # Scale input
        arr_scaled = scaler.transform(arr)

        # Prediction
        prob = float(model.predict_proba(arr_scaled)[0][1])
        pred = int(prob >= 0.5)

        return jsonify({
            'prediction': pred,
            'probability': round(prob, 4),
            'risk': 'high' if prob >= 0.6 else 'mid' if prob >= 0.4 else 'low'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ✅ Run locally
if __name__ == '__main__':
    app.run(debug=True)
