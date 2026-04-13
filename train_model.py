import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

INPUT_FEATURES = [
    "perimeter_worst",
    "area_worst",
    "concave points_worst",
    "concave points_mean",
    "radius_worst",
    "radius_mean",
    "perimeter_mean",
    "area_mean",
    "concavity_mean",
    "concavity_worst",
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


def main():
    df = pd.read_csv("breast-cancer.csv")

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])

    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    if df["diagnosis"].isnull().any():
        raise ValueError("Diagnosis column contains unexpected values. Expected only M and B.")

    missing = [col for col in INPUT_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    X_base = df[INPUT_FEATURES].copy()
    y = df["diagnosis"].copy()

    X_model = add_engineered_features(X_base)
    model_features = X_model.columns.tolist()

    feature_ranges = {}
    for col in INPUT_FEATURES:
        feature_ranges[col] = {
            "min": float(X_base[col].min()),
            "max": float(X_base[col].max()),
            "mean": float(X_base[col].mean()),
        }

    X_train_base, X_test_base, y_train, y_test = train_test_split(
        X_base,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train = add_engineered_features(X_train_base)
    X_test = add_engineered_features(X_test_base)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
    }

    comparison = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        comparison[name] = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        }
        trained_models[name] = model

    best_model_name = max(
        comparison,
        key=lambda name: (comparison[name]["f1"], comparison[name]["accuracy"])
    )
    best_model = trained_models[best_model_name]

    explanation_model = RandomForestClassifier(n_estimators=300, random_state=42)
    explanation_model.fit(X_train_scaled, y_train)
    feature_importances = {
        feature: float(importance)
        for feature, importance in zip(model_features, explanation_model.feature_importances_)
    }

    joblib.dump(best_model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(INPUT_FEATURES, "input_features.pkl")
    joblib.dump(model_features, "model_features.pkl")
    joblib.dump(feature_importances, "feature_importances.pkl")
    joblib.dump(comparison, "model_comparison.pkl")
    joblib.dump(feature_ranges, "feature_ranges.pkl")
    joblib.dump(best_model_name, "best_model_name.pkl")

    print("Training completed successfully.")
    print(f"Best model selected: {best_model_name}")
    print("Saved all model artifacts.")


if __name__ == "__main__":
    main()