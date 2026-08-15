# Heart Disease Classifier Interface (Flask + Streamlit)

## ⚠️ First: retrain your model, it has a data leakage bug

Your original notebook never dropped the `Heart_Disease` target column from the
features, so the model was trained with the answer included as an input. That's
why you saw 100% accuracy — it wasn't real.

1. Put `synthetic_heart_disease_dataset.csv` in this folder.
2. Run:
   ```
   python train_model_fixed.py
   ```
   This retrains a clean `LogisticRegression` model (without the leaked column),
   prints a realistic accuracy + classification report, and saves `model.pkl`
   in this folder.

## Setup

1. Make sure the retrained `model.pkl` is in this folder (the script above saves it here automatically).
2. `MODEL_PATH` in `flask_api.py` is already set to `"model.pkl"` — update it only if you rename the file.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Run it

**Terminal 1 — start the API:**
```
python flask_api.py
```
This runs on `http://localhost:5000`.

**Terminal 2 — start the UI:**
```
streamlit run streamlit_app.py
```
This opens in your browser, usually at `http://localhost:8501`.

## How it works

- `flask_api.py` loads your model once and exposes:
  - `GET /health` — check it's running
  - `GET /features` — lists expected input columns (if your model was trained on a pandas DataFrame, sklearn saves these automatically)
  - `POST /predict` — takes feature values, returns a prediction
- `streamlit_app.py` is the UI. It has two modes:
  - **Single prediction** — fill in one row of values by hand
  - **Batch prediction** — upload a CSV and get predictions for every row, downloadable as a new CSV

## What features the model expects

After the fix, the model is trained on these 14 columns (everything except
`Smoking`, `Alcohol_Intake`, `Physical_Activity`, `Diet`, `Stress_Level`,
`Gender`, and the target `Heart_Disease`):

```
Age, Weight, Height, BMI, Hypertension, Diabetes, Hyperlipidemia,
Family_History, Previous_Heart_Attack, Systolic_BP, Diastolic_BP,
Heart_Rate, Blood_Sugar_Fasting, Cholesterol_Total
```

Since your model was trained on a pandas DataFrame, sklearn stores these
names automatically (`feature_names_in_`), so the Streamlit form will
auto-generate one input box per feature — no manual setup needed.

## Notes

- The prediction is `0` (no heart disease) or `1` (heart disease), shown alongside the model's confidence for each class.
- If you want to also use the categorical columns you dropped (`Smoking`, `Diet`, etc.), you'll need to encode them numerically (e.g. one-hot encoding) before training — LogisticRegression can't take raw text columns.
