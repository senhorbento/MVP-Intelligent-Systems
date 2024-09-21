from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive the data from the JSON request
        data = request.json

        # Check for the required fields
        if 'temp' not in data or 'umidade' not in data or 'chuva' not in data:
            return jsonify({'error': 'Missing fields: temp, umidade, chuva'}), 400

        # Extract the correct features for the model
        input_features = pd.DataFrame({
            'Avg_Temperature': [data['temp']],
            'Max_Humidity': [data['umidade']],
            'Max_Preciptation': [data['chuva']],
        })

        # Make the prediction using the trained model
        prediction = model.predict(input_features)

        # Return the prediction
        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
