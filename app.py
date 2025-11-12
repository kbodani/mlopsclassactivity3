from flask import Flask, request, jsonify
import pickle
import numpy as np
import glob
import os

app = Flask(__name__)

# Load the latest model
model_files = glob.glob("model_v*.pkl")
if not model_files:
    raise FileNotFoundError("No model file found!")

latest_model = max(model_files, key=os.path.getctime)
print(f"Loading model: {latest_model}")

with open(latest_model, "rb") as f:
    model = pickle.load(f)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": latest_model
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint
    
    Example request:
    {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    """
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        # Iris class names
        class_names = ['setosa', 'versicolor', 'virginica']
        
        return jsonify({
            "prediction": int(prediction[0]),
            "class": class_names[int(prediction[0])],
            "probabilities": {
                class_names[i]: float(probability[0][i]) 
                for i in range(len(class_names))
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        "model_file": latest_model,
        "n_estimators": model.n_estimators,
        "n_features": model.n_features_in_,
        "n_classes": len(model.classes_)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)