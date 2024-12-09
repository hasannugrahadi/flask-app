from flask import Flask, jsonify
import numpy as np
import pickle
import requests

app = Flask(__name__)

with open("perceptron_model.pkl", "rb") as model_file:
    loaded_weights, loaded_bias = pickle.load(model_file)

def step_function(x):
    return np.where(x >= 0, 1, 0)

def perceptron_predict(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    return step_function(linear_output)

@app.route('/api/predict_status', methods=['GET'])
def predict_status():
    external_url = "https://farmbro-mbkm.research-ai.my.id/api/gas"
    response = requests.get(external_url)

    if response.status_code != 200:
        return jsonify({
            "error": "Failed to fetch data", 
            "status_code": response.status_code}
            ), 500

    json_data = response.json()
    data = json_data.get("data", {})
    temperature = data.get("temperature", 0)
    humidity = data.get("humidity", 0)
    ammonia = data.get("amonia", 0)

    temperature = 0 if 20 < temperature < 30 else 1
    humidity = 0 if 50 < humidity < 70 else 1
    ammonia = 0 if ammonia < 25 else 1

    X = np.array([temperature, humidity, ammonia])

    prediction = perceptron_predict(X, loaded_weights, loaded_bias)
    status = 'Baik' if prediction == 0 else 'Buruk'

    return jsonify({
        "status": "success",
        "prediction": status
    })

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=5000)
