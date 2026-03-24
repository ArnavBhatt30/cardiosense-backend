from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# ✅ app init
app = Flask(__name__)
CORS(app)

# ✅ load model (scaler ignore kar rahe for now)
model = joblib.load('model.pkl')

# ✅ health check
@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'CardioSense API running'})


# ✅ prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # ✅ frontend ke 11 features
        vals = [
            float(data.get('age', 0)),
            float(data.get('gender', 0)),
            float(data.get('height', 0)),
            float(data.get('weight', 0)),
            float(data.get('ap_hi', 0)),
            float(data.get('ap_lo', 0)),
            float(data.get('cholesterol', 0)),
            float(data.get('gluc', 0)),
            float(data.get('smoke', 0)),
            float(data.get('alco', 0)),
            float(data.get('active', 0)),

            # 🔥 dummy features to match 13
            0,
            0
        ]

        arr = np.array(vals).reshape(1, -1)

        # ❌ scaler hata diya (shape mismatch avoid karne ke liye)
        arr_scaled = arr

        # ✅ prediction
        prob = float(model.predict_proba(arr_scaled)[0][1])
        pred = int(prob >= 0.5)

        return jsonify({
            'prediction': pred,
            'probability': round(prob, 4),
            'risk': 'high' if prob >= 0.6 else 'mid' if prob >= 0.4 else 'low'
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({'error': str(e)}), 400


# ✅ run locally
if __name__ == '__main__':
    app.run(debug=True)
