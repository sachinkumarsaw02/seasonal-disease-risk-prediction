import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "dataset", "disease_data.csv")
MODEL_DIR = os.path.join(BASE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA)

features = ["Location", "Area", "Month", "Temperature", "Humidity", "Rainfall"]
target = "Disease_Risk"

X = df[features]
y = df[target]

categorical = ["Location", "Area"]
numeric = ["Month", "Temperature", "Humidity", "Rainfall"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric)
    ]
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("=" * 55)
print("SEASONAL DISEASE RISK MODEL")
print("=" * 55)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
print("=" * 55)

joblib.dump(pipeline, os.path.join(MODEL_DIR, "disease_model.pkl"))
print("Saved: model/disease_model.pkl")
