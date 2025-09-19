# train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Dummy hospital dataset
data = {
    "age": [25, 40, 60, 30, 45, 70, 55, 35],
    "bp": [120, 140, 150, 130, 135, 160, 145, 128],
    "cholesterol": [180, 220, 240, 200, 210, 260, 230, 190],
    "readmitted": [0, 1, 1, 0, 1, 1, 1, 0]  # target
}
df = pd.DataFrame(data)

X = df[["age", "bp", "cholesterol"]]
y = df["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")
