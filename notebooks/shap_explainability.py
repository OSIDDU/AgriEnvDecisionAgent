import os
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

PROC_DIR = "data/processed"
MODEL_DIR = "models"

def explain(model_file, data_file, title):
    print(f"\nGenerating SHAP for {title} ...")

    model = joblib.load(model_file)
    data = pd.read_csv(data_file)
    X = data.iloc[:, :-1]

    # Tree-based explainer (Random Forest optimized)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary Plot
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"{title} - SHAP Summary")
    plt.tight_layout()
    plt.savefig(f"{title}_shap_summary.png")
    plt.show()

    # Bar Plot
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f"{title} - Feature Impact")
    plt.tight_layout()
    plt.savefig(f"{title}_shap_bar.png")
    plt.show()

    print("SHAP plots saved.")


if __name__ == "__main__":
    explain(
        model_file=os.path.join(MODEL_DIR, "yield_model.pkl"),
        data_file=os.path.join(PROC_DIR, "crop_clean.csv"),
        title="Crop_Yield"
    )

    explain(
        model_file=os.path.join(MODEL_DIR, "aqi_model.pkl"),
        data_file=os.path.join(PROC_DIR, "aqi_clean.csv"),
        title="Air_Quality"
    )
