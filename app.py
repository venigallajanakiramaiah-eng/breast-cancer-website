import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

ARTIFACT_FILES = [
    "model.pkl",
    "scaler.pkl",
    "input_features.pkl",
    "model_features.pkl",
    "feature_importances.pkl",
    "model_comparison.pkl",
    "feature_ranges.pkl",
    "best_model_name.pkl",
]

EPSILON = 1e-6


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["area_perimeter_ratio_mean"] = out["area_mean"] / (out["perimeter_mean"] + EPSILON)
    out["area_perimeter_ratio_worst"] = out["area_worst"] / (out["perimeter_worst"] + EPSILON)
    out["radius_gap"] = out["radius_worst"] - out["radius_mean"]
    out["concavity_score"] = out["concavity_mean"] + out["concavity_worst"]
    out["concave_points_gap"] = out["concave points_worst"] - out["concave points_mean"]

    return out


def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    input_features = joblib.load("input_features.pkl")
    model_features = joblib.load("model_features.pkl")
    feature_importances = joblib.load("feature_importances.pkl")
    model_comparison = joblib.load("model_comparison.pkl")
    feature_ranges = joblib.load("feature_ranges.pkl")
    best_model_name = joblib.load("best_model_name.pkl")

    return {
        "model": model,
        "scaler": scaler,
        "input_features": input_features,
        "model_features": model_features,
        "feature_importances": feature_importances,
        "model_comparison": model_comparison,
        "feature_ranges": feature_ranges,
        "best_model_name": best_model_name,
    }


def files_ready() -> bool:
    return all(os.path.exists(file_name) for file_name in ARTIFACT_FILES)


def to_label(text: str) -> str:
    return text.replace("_", " ").replace("  ", " ").title()


def get_sample_data():
    return {
        "Benign Sample": {
            "perimeter_worst": 85.0,
            "area_worst": 500.0,
            "concave points_worst": 0.05,
            "concave points_mean": 0.03,
            "radius_worst": 13.0,
            "radius_mean": 11.5,
            "perimeter_mean": 75.0,
            "area_mean": 420.0,
            "concavity_mean": 0.04,
            "concavity_worst": 0.08,
        },
        "Malignant Sample": {
            "perimeter_worst": 140.0,
            "area_worst": 1300.0,
            "concave points_worst": 0.20,
            "concave points_mean": 0.12,
            "radius_worst": 25.0,
            "radius_mean": 18.0,
            "perimeter_mean": 120.0,
            "area_mean": 1000.0,
            "concavity_mean": 0.25,
            "concavity_worst": 0.45,
        },
        "Borderline Sample": {
            "perimeter_worst": 100.0,
            "area_worst": 750.0,
            "concave points_worst": 0.10,
            "concave points_mean": 0.06,
            "radius_worst": 16.0,
            "radius_mean": 14.0,
            "perimeter_mean": 90.0,
            "area_mean": 650.0,
            "concavity_mean": 0.10,
            "concavity_worst": 0.18,
        },
    }


def validate_inputs(form_data, input_features, feature_ranges):
    errors = []
    values = {}

    for feature in input_features:
        raw_value = form_data.get(feature, "").strip()

        if raw_value == "":
            errors.append(f"{to_label(feature)} is required.")
            continue

        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"{to_label(feature)} must be a valid number.")
            continue

        if value < 0:
            errors.append(f"{to_label(feature)} cannot be negative.")
            continue

        observed_min = feature_ranges[feature]["min"]
        observed_max = feature_ranges[feature]["max"]

        lower_bound = max(0.0, observed_min * 0.8)
        upper_bound = observed_max * 1.2

        if value < lower_bound or value > upper_bound:
            errors.append(
                f"{to_label(feature)} looks out of realistic range "
                f"(expected about {observed_min:.4f} to {observed_max:.4f})."
            )
            continue

        values[feature] = value

    return errors, values


@app.route("/")
def home():
    if not files_ready():
        return render_template(
            "index.html",
            files_ready=False,
            prediction_text="Run python train_model.py first.",
            sample_data=get_sample_data(),
        )

    artifacts = load_artifacts()

    return render_template(
        "index.html",
        files_ready=True,
        features=artifacts["input_features"],
        labels={feature: to_label(feature) for feature in artifacts["input_features"]},
        model_comparison=artifacts["model_comparison"],
        best_model_name=artifacts["best_model_name"],
        sample_data=get_sample_data(),
        form_values={},
    )


@app.route("/predict", methods=["POST"])
def predict():
    if not files_ready():
        return render_template(
            "index.html",
            files_ready=False,
            prediction_text="Run python train_model.py first.",
            sample_data=get_sample_data(),
        )

    artifacts = load_artifacts()

    input_features = artifacts["input_features"]
    model_features = artifacts["model_features"]
    feature_ranges = artifacts["feature_ranges"]

    errors, values = validate_inputs(request.form, input_features, feature_ranges)

    if errors:
        return render_template(
            "index.html",
            files_ready=True,
            features=input_features,
            labels={feature: to_label(feature) for feature in input_features},
            model_comparison=artifacts["model_comparison"],
            best_model_name=artifacts["best_model_name"],
            sample_data=get_sample_data(),
            errors=errors,
            form_values=request.form,
        )

    base_df = pd.DataFrame([values])
    model_df = add_engineered_features(base_df)
    model_df = model_df[model_features]

    input_scaled = artifacts["scaler"].transform(model_df)
    model = artifacts["model"]

    prediction = int(model.predict(input_scaled)[0])
    probability = model.predict_proba(input_scaled)[0]
    malignant_probability = float(probability[1]) * 100

    if prediction == 1:
        prediction_text = "Prediction: Malignant"
        confidence = round(malignant_probability, 2)
    else:
        prediction_text = "Prediction: Benign"
        confidence = round(100 - malignant_probability, 2)

    if malignant_probability >= 80:
        risk_level = "High Risk"
        risk_class = "high"
    elif malignant_probability >= 50:
        risk_level = "Moderate Risk"
        risk_class = "moderate"
    else:
        risk_level = "Low Risk"
        risk_class = "low"

    # Simple prediction explanation using standardized magnitude × feature importance
    importance_series = pd.Series(artifacts["feature_importances"])
    scaled_series = pd.Series(np.abs(input_scaled[0]), index=model_features)
    contribution_scores = (importance_series * scaled_series).sort_values(ascending=False)
    top_factors = [to_label(name) for name in contribution_scores.head(3).index.tolist()]

    return render_template(
        "index.html",
        files_ready=True,
        features=input_features,
        labels={feature: to_label(feature) for feature in input_features},
        model_comparison=artifacts["model_comparison"],
        best_model_name=artifacts["best_model_name"],
        sample_data=get_sample_data(),
        form_values=values,
        prediction_text=prediction_text,
        confidence_text=f"Confidence: {confidence}%",
        malignant_probability=f"{malignant_probability:.2f}%",
        risk_level=risk_level,
        risk_class=risk_class,
        top_factors=top_factors,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)