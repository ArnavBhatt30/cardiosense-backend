features = [
    'age','gender','height','weight',
    'ap_hi','ap_lo','cholesterol','gluc',
    'smoke','alco','active'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        vals = [float(data[f]) for f in features]

        arr = np.array(vals).reshape(1, -1)
        arr_scaled = scaler.transform(arr)

        prob = float(model.predict_proba(arr_scaled)[0][1])
        pred = int(prob >= 0.5)

        return jsonify({
            'prediction': pred,
            'probability': round(prob, 4),
            'risk': 'high' if prob >= 0.6 else 'mid' if prob >= 0.4 else 'low'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400
