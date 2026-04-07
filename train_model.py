import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("breast-cancer.csv")

print("First 5 rows of dataset:")
print(df.head())

# Drop unnecessary column
if "id" in df.columns:
    df = df.drop("id", axis=1)

# Remove unnamed column if present
if "Unnamed: 32" in df.columns:
    df = df.drop("Unnamed: 32", axis=1)

# Convert diagnosis column from M/B to 1/0
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Check conversion
print("\nUnique values in diagnosis after mapping:")
print(df["diagnosis"].unique())

# Features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train temporary model to find important features
temp_model = RandomForestClassifier(n_estimators=200, random_state=42)
temp_model.fit(X_train, y_train)

# Select top 10 important features
feature_importances = pd.Series(temp_model.feature_importances_, index=X.columns)
selected_features = feature_importances.sort_values(ascending=False).head(10).index.tolist()

print("\nSelected Features:")
for i, feature in enumerate(selected_features, start=1):
    print(f"{i}. {feature}")

# Keep only selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Train final model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model files
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selected_features, "selected_features.pkl")

print("\nFiles saved successfully:")
print("- model.pkl")
print("- scaler.pkl")
print("- selected_features.pkl")