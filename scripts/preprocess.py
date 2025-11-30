# Preprocessing script
import os
import pandas as pd

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
os.makedirs(PROC_DIR, exist_ok=True)


def preprocess_crop():
    path = os.path.join(RAW_DIR, "crop_yield.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Place your crop dataset there.")

    print(f"[CROP] Reading: {path}")
    df = pd.read_csv(path)

    df = df.loc[:, ~df.columns.duplicated()]          # drop duplicate columns
    df = df[df.isna().mean(axis=1) < 0.5].copy()      # drop rows with >50% NaN

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        raise ValueError("Not enough numeric columns in crop dataset.")

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    ordered_cols = num_cols                       # last one is assumed target
    df_out = df[ordered_cols].copy()
    out_path = os.path.join(PROC_DIR, "crop_clean.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[CROP] Saved cleaned crop data to {out_path}")
    print(f"[CROP] Features: {ordered_cols[:-1]} | Target: {ordered_cols[-1]}")


def preprocess_aqi():
    path = os.path.join(RAW_DIR, "air_quality.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Place your AQI dataset there.")

    print(f"[AQI] Reading: {path}")
    df = pd.read_csv(path)

    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(axis=1, how="all")   # remove fully empty columns

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        raise ValueError("Not enough numeric columns in AQI dataset.")

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    ordered_cols = num_cols
    df_out = df[ordered_cols].copy()
    out_path = os.path.join(PROC_DIR, "aqi_clean.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[AQI] Saved cleaned AQI data to {out_path}")
    print(f"[AQI] Features: {ordered_cols[:-1]} | Target: {ordered_cols[-1]}")


if __name__ == "__main__":
    preprocess_crop()
    preprocess_aqi()
