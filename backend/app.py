from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask("SuperKart")
MODEL_PATH = Path(__file__).resolve().parent / "superkart_model.joblib"
FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
    "Store_Age_Years", "Product_Type_Category"
]
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

def get_model():
    if model is None:
        return None
    return model

@superkart_api.get("/")
def home():
    return jsonify({"service": "SuperKart Sales Prediction API", "status": "ok", "model_loaded": get_model() is not None})

@superkart_api.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": get_model() is not None})

@superkart_api.post("/v1/predict")
def predict_sales():
    predictor = get_model()
    if predictor is None:
        return jsonify({"error": "Model artifact not found. Run the notebook first."}), 503
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    missing = [c for c in FEATURES if c not in data]
    if missing:
        return jsonify({"error": "Missing required features", "missing_features": missing}), 400
    sample = pd.DataFrame([{c: data[c] for c in FEATURES}], columns=FEATURES)
    try:
        prediction = float(predictor.predict(sample)[0])
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 400
    return jsonify({"Sales": round(prediction, 2), "target": "Product_Store_Sales_Total"})

@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    predictor = get_model()
    if predictor is None:
        return jsonify({"error": "Model artifact not found. Run the notebook first."}), 503
    if "file" not in request.files:
        return jsonify({"error": "Upload a CSV using the 'file' field."}), 400
    try:
        input_data = pd.read_csv(request.files["file"])
    except Exception as exc:
        return jsonify({"error": f"Could not read CSV: {exc}"}), 400
    missing = [c for c in FEATURES if c not in input_data.columns]
    if missing:
        return jsonify({"error": "CSV is missing required features", "missing_features": missing}), 400
    try:
        predictions = predictor.predict(input_data[FEATURES])
    except Exception as exc:
        return jsonify({"error": f"Batch prediction failed: {exc}"}), 400
    output = input_data.copy()
    output["Predicted_Product_Store_Sales_Total"] = np.round(predictions, 2)
    return jsonify({"predictions": output.to_dict(orient="records")})

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
