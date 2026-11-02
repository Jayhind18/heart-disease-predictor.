"""
Corrected training script - fixes the data leakage bug found in your notebook.

The bug: 'Heart_Disease' (the target) was never dropped from x, so the model
was literally given the answer as an input feature. That's why accuracy was 100%.

Run this to retrain a clean model and save it as model.pkl
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
data = pd.read_csv("synthetic_heart_disease_dataset.csv")
data["Alcohol_Intake"] = data["Alcohol_Intake"].fillna("unknown")

# ------------------------------------------------------------------
# 2. Build features / target -- THIS IS THE FIX
#    Added 'Heart_Disease' to the drop list so it's no longer a feature
# ------------------------------------------------------------------
x = data.drop(
    ["Smoking", "Alcohol_Intake", "Physical_Activity", "Diet",
     "Stress_Level", "Gender", "Heart_Disease"],
    axis=1
)
y = data["Heart_Disease"]

print("Features used for training:", list(x.columns))

# ------------------------------------------------------------------
# 3. Train / test split
# ------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# 4. Train model
# ------------------------------------------------------------------
model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)

# ------------------------------------------------------------------
# 5. Evaluate (should now be realistic, not 100%)
# ------------------------------------------------------------------
y_pred = model.predict(x_test)
probabilities = model.predict_proba(x_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------------
# 6. Save the model with joblib (matches flask_api.py's joblib.load)
# ------------------------------------------------------------------
joblib.dump(model, "model.pkl")
print("\nSaved clean model to model.pkl")
