from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
FEATURES_FILE = "selected_features.pkl"


def load_artifacts():
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    selected_features = joblib.load(FEATURES_FILE)
    return model, scaler, selected_features


@app.route("/")
def home():
    files_ready = all(
        os.path.exists(file_name)
        for file_name in [MODEL_FILE, SCALER_FILE, FEATURES_FILE]
    )

    if files_ready:
        _, _, selected_features = load_artifacts()
        return render_template(
            "index.html",
            features=selected_features,
            files_ready=True
        )

    return render_template(
        "index.html",
        features=[],
        files_ready=False,
        prediction_text="Please run train_model.py first to generate model files."
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        model, scaler, selected_features = load_artifacts()

        input_values = []
        for feature in selected_features:
            value = float(request.form[feature])
            input_values.append(value)

        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        if prediction == 1:
            result = "Malignant"
            confidence = round(probability[1] * 100, 2)
        else:
            result = "Benign"
            confidence = round(probability[0] * 100, 2)

        return render_template(
            "index.html",
            features=selected_features,
            files_ready=True,
            prediction_text=f"Prediction: {result}",
            confidence_text=f"Confidence: {confidence}%"
        )

    except Exception as e:
        return render_template(
            "index.html",
            features=[],
            files_ready=False,
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)