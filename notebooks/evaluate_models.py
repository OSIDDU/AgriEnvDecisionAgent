import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

PROC_DIR = "data/processed"
MODEL_DIR = "models"


def evaluate(model_path, data_path, title):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    preds = model.predict(X)

    # Metrics
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    print(f"\n===== {title} Evaluation =====")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    # -------- Feature Importance -------- #
    plt.figure(figsize=(6, 4))
    importances = model.feature_importances_
    plt.barh(X.columns, importances)
    plt.xlabel("Importance")
    plt.title(f"{title} - Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{title}_feature_importance.png")
    plt.show()

    # -------- Actual vs Predicted -------- #
    plt.figure(figsize=(6, 4))
    plt.plot(y.values, label="Actual")
    plt.plot(preds, label="Predicted")
    plt.title(f"{title} - Actual vs Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title}_actual_vs_predicted.png")
    plt.show()


if __name__ == "__main__":
    evaluate(
        model_path=os.path.join(MODEL_DIR, "yield_model.pkl"),
        data_path=os.path.join(PROC_DIR, "crop_clean.csv"),
        title="Crop_Yield"
    )

    evaluate(
        model_path=os.path.join(MODEL_DIR, "aqi_model.pkl"),
        data_path=os.path.join(PROC_DIR, "aqi_clean.csv"),
        title="Air_Quality"
    )

