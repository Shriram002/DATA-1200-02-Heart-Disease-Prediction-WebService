from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder="templates")

# Load Saved Models

with open("random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("kmeans_model.pkl", "rb") as f:
    kmeans_model = pickle.load(f)

# Load the feature names saved during training (exact order is required)
with open("feature_names.pkl", "rb") as f:
    feature_columns = pickle.load(f)

print("Loaded Feature Columns:", feature_columns)

def parse_input(json_data):
    """
    Converts incoming JSON into a pandas DataFrame with columns in the order of feature_columns.
    This ensures that the input has valid feature names.
    """
    try:
        data = {col: [float(json_data[col])] for col in feature_columns}
        return pd.DataFrame(data, columns=feature_columns)
    except Exception as e:
        print("Parsing Error:", e)
        return None

# API Endpoints

@app.route("/")
def home():
    return render_template("index.html") 

@app.route("/randomforest/evaluate", methods=["POST"])
def evaluate_rf():
    data = request.get_json(force=True)
    features_input = parse_input(data)
    if features_input is None:
        return jsonify({"error": "Invalid input format."}), 400
    prediction = rf_model.predict(features_input)
    # For supervised models, prediction is binary.
    diagnosis = "Heart Disease Detected. Please consult with your doctor." if int(prediction[0]) == 1 else "No Heart Disease Detected."
    return jsonify({"model": "Random Forest", "prediction": int(prediction[0]), "diagnosis": diagnosis})

@app.route("/svm/evaluate", methods=["POST"])
def evaluate_svm():
    data = request.get_json(force=True)
    features_input = parse_input(data)
    if features_input is None:
        return jsonify({"error": "Invalid input format."}), 400
    prediction = svm_model.predict(features_input)
    diagnosis = "Heart Disease Detected. Please consult with your doctor." if int(prediction[0]) == 1 else "No Heart Disease Detected."
    return jsonify({"model": "SVM", "prediction": int(prediction[0]), "diagnosis": diagnosis})

@app.route("/kmeans/evaluate", methods=["POST"])
def evaluate_kmeans():
    data = request.get_json(force=True)
    features_input = parse_input(data)
    if features_input is None:
        return jsonify({"error": "Invalid input format."}), 400
    # Convert the DataFrame to a NumPy array for KMeans
    cluster = int(kmeans_model.predict(features_input.to_numpy())[0])
    
    # Mapping based on your cluster analysis:
 
    if cluster == 0:
        diagnosis = "No Heart Disease Detected."
        binary = 0
    else:
        diagnosis = "Heart Disease Detected. Please consult with your doctor."
        binary = 1
        
    return jsonify({"model": "K-means", "cluster": cluster, "prediction": binary, "diagnosis": diagnosis})

if __name__ == "__main__":
    app.run(debug=True)
