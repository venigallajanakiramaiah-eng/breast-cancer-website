import os
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet

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
    return {
        "model": joblib.load("model.pkl"),
        "scaler": joblib.load("scaler.pkl"),
        "input_features": joblib.load("input_features.pkl"),
        "model_features": joblib.load("model_features.pkl"),
        "feature_importances": joblib.load("feature_importances.pkl"),
        "model_comparison": joblib.load("model_comparison.pkl"),
        "feature_ranges": joblib.load("feature_ranges.pkl"),
        "best_model_name": joblib.load("best_model_name.pkl"),
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


def build_chart_data(values, feature_ranges, feature_importances):
    input_feature_names = list(values.keys())

    avg_labels = [to_label(f) for f in input_feature_names]
    patient_values = [float(values[f]) for f in input_feature_names]
    average_values = [float(feature_ranges[f]["mean"]) for f in input_feature_names]

    radar_patient = []
    radar_average = []
    local_influence_scores = []

    for feature in input_feature_names:
        min_v = float(feature_ranges[feature]["min"])
        max_v = float(feature_ranges[feature]["max"])
        mean_v = float(feature_ranges[feature]["mean"])
        user_v = float(values[feature])

        denominator = max(max_v - min_v, 1e-6)

        patient_scaled = ((user_v - min_v) / denominator) * 100
        average_scaled = ((mean_v - min_v) / denominator) * 100

        radar_patient.append(round(patient_scaled, 2))
        radar_average.append(round(average_scaled, 2))

        distance_from_average = abs(user_v - mean_v) / denominator
        importance_weight = float(feature_importances.get(feature, 0.0))
        influence_score = distance_from_average * importance_weight
        local_influence_scores.append((feature, influence_score))

    local_influence_scores = sorted(
        local_influence_scores,
        key=lambda item: item[1],
        reverse=True
    )[:6]

    importance_labels = [to_label(k) for k, _ in local_influence_scores]
    importance_values = [round(float(v), 4) for _, v in local_influence_scores]

    return {
        "importance_labels": importance_labels,
        "importance_values": importance_values,
        "avg_labels": avg_labels,
        "radar_patient": radar_patient,
        "radar_average": radar_average,
        "raw_patient_values": patient_values,
        "raw_average_values": average_values,
    }


def compute_prediction_bundle(values, artifacts):
    input_features = artifacts["input_features"]
    model_features = artifacts["model_features"]

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

    top_factor_scores = []

    for feature in input_features:
        min_v = float(artifacts["feature_ranges"][feature]["min"])
        max_v = float(artifacts["feature_ranges"][feature]["max"])
        mean_v = float(artifacts["feature_ranges"][feature]["mean"])
        user_v = float(values[feature])

        denominator = max(max_v - min_v, 1e-6)
        distance_from_average = abs(user_v - mean_v) / denominator
        importance_weight = float(artifacts["feature_importances"].get(feature, 0.0))
        score = distance_from_average * importance_weight
        top_factor_scores.append((feature, score))

    top_factor_scores = sorted(top_factor_scores, key=lambda x: x[1], reverse=True)
    top_factors = [to_label(name) for name, _ in top_factor_scores[:3]]

    chart_data = build_chart_data(
        values=values,
        feature_ranges=artifacts["feature_ranges"],
        feature_importances=artifacts["feature_importances"],
    )

    return {
        "prediction_text": prediction_text,
        "confidence": confidence,
        "malignant_probability": round(malignant_probability, 2),
        "risk_level": risk_level,
        "risk_class": risk_class,
        "top_factors": top_factors,
        "chart_data": chart_data,
    }


def create_risk_gauge_image(malignant_probability):
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={"aspect": "equal"})
    ax.axis("off")

    color = "#16a34a"
    if malignant_probability >= 80:
        color = "#dc2626"
    elif malignant_probability >= 50:
        color = "#f59e0b"

    sizes = [malignant_probability, 100 - malignant_probability]
    ax.pie(
        sizes,
        startangle=180,
        counterclock=False,
        colors=[color, "#e5e7eb"],
        wedgeprops={"width": 0.35, "edgecolor": "white"},
    )

    circle = plt.Circle((0, 0), 0.45, color="white")
    ax.add_artist(circle)

    ax.text(0, 0.05, f"{malignant_probability:.1f}%", ha="center", va="center",
            fontsize=18, fontweight="bold")
    ax.text(0, -0.18, "Risk Meter", ha="center", va="center",
            fontsize=10, color="#475569")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.15, 1.1)

    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buffer.seek(0)
    return buffer


def create_feature_importance_image(chart_data):
    labels = chart_data["importance_labels"]
    values = chart_data["importance_values"]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={"projection": "polar"})
    theta = np.linspace(0.0, 2 * np.pi, len(values), endpoint=False)
    widths = np.repeat((2 * np.pi) / len(values), len(values))

    colors_list = ["#3b82f6", "#0ea5e9", "#6366f1", "#a855f7", "#ec4899", "#10b981"]
    bars = ax.bar(theta, values, width=widths, bottom=0.0, alpha=0.8)

    for i, bar in enumerate(bars):
        bar.set_facecolor(colors_list[i % len(colors_list)])
        bar.set_edgecolor("white")
        bar.set_linewidth(1.5)

    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Feature Influence", pad=20, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25)

    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def create_radar_chart_image(chart_data):
    labels = chart_data["avg_labels"]
    patient = chart_data["radar_patient"]
    average = chart_data["radar_average"]

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    patient_plot = patient + patient[:1]
    average_plot = average + average[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax.plot(angles, patient_plot, linewidth=2, label="Patient")
    ax.fill(angles, patient_plot, alpha=0.20)
    ax.plot(angles, average_plot, linewidth=2, label="Average")
    ax.fill(angles, average_plot, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels([])
    ax.set_ylim(0, 100)
    ax.set_title("Input vs Average Comparison", pad=20, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.10))

    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def create_prediction_pdf(values, bundle):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Breast Cancer Prediction Report", styles["Title"]))
    story.append(Spacer(1, 12))

    summary_data = [
        ["Prediction", bundle["prediction_text"].replace("Prediction: ", "")],
        ["Confidence", f'{bundle["confidence"]}%'],
        ["Malignant Probability", f'{bundle["malignant_probability"]}%'],
        ["Risk Level", bundle["risk_level"]],
        ["Top Contributing Factors", ", ".join(bundle["top_factors"])],
    ]

    summary_table = Table(summary_data, colWidths=[170, 320])
    summary_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf2ff")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 18))

    story.append(Paragraph("User Input Data", styles["Heading2"]))
    input_rows = [["Feature", "Value"]]
    for feature, value in values.items():
        input_rows.append([to_label(feature), str(value)])

    input_table = Table(input_rows, colWidths=[250, 240])
    input_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f4c81")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(input_table)
    story.append(Spacer(1, 18))

    story.append(Paragraph("Visual Analysis", styles["Heading2"]))
    story.append(Spacer(1, 8))

    risk_img = Image(create_risk_gauge_image(bundle["malignant_probability"]), width=3.0 * inch, height=1.9 * inch)
    feature_img = Image(create_feature_importance_image(bundle["chart_data"]), width=3.0 * inch, height=3.0 * inch)
    radar_img = Image(create_radar_chart_image(bundle["chart_data"]), width=5.6 * inch, height=4.4 * inch)

    top_images = Table([[risk_img, feature_img]], colWidths=[250, 250])
    story.append(top_images)
    story.append(Spacer(1, 12))
    story.append(radar_img)

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer


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
        chart_data=None,
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
            chart_data=None,
        )

    bundle = compute_prediction_bundle(values, artifacts)

    return render_template(
        "index.html",
        files_ready=True,
        features=input_features,
        labels={feature: to_label(feature) for feature in input_features},
        model_comparison=artifacts["model_comparison"],
        best_model_name=artifacts["best_model_name"],
        sample_data=get_sample_data(),
        form_values=values,
        prediction_text=bundle["prediction_text"],
        confidence_text=f'Confidence: {bundle["confidence"]}%',
        malignant_probability=bundle["malignant_probability"],
        risk_level=bundle["risk_level"],
        risk_class=bundle["risk_class"],
        top_factors=bundle["top_factors"],
        chart_data=bundle["chart_data"],
    )


@app.route("/download-report", methods=["POST"])
def download_report():
    if not files_ready():
        return "Run python train_model.py first.", 400

    artifacts = load_artifacts()
    input_features = artifacts["input_features"]
    feature_ranges = artifacts["feature_ranges"]

    errors, values = validate_inputs(request.form, input_features, feature_ranges)
    if errors:
        return "Invalid input data. Please go back and correct the values.", 400

    bundle = compute_prediction_bundle(values, artifacts)
    pdf_buffer = create_prediction_pdf(values, bundle)

    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name="breast_cancer_prediction_report.pdf",
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 50033))
    app.run(host="0.0.0.0", port=port, debug=False)