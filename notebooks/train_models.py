import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

PROC_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_yield_model():
    path = os.path.join(PROC_DIR, "crop_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run scripts/preprocess.py first.")

    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    out_path = os.path.join(MODEL_DIR, "yield_model.pkl")
    joblib.dump(model, out_path)

    print(f"[YIELD] MAE = {mae:.4f}, R² = {r2:.4f}")
    print(f"[YIELD] Model saved to {out_path}")


def train_aqi_model():
    path = os.path.join(PROC_DIR, "aqi_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run scripts/preprocess.py first.")

    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    out_path = os.path.join(MODEL_DIR, "aqi_model.pkl")
    joblib.dump(model, out_path)

    print(f"[AQI] MAE = {mae:.4f}, R² = {r2:.4f}")
    print(f"[AQI] Model saved to {out_path}")


if __name__ == "__main__":
    train_yield_model()
    train_aqi_model()
