import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# Optional Kaggle download
try:
    from kagglehub import dataset_download
    KAGGLE_AVAILABLE = True
except Exception:
    KAGGLE_AVAILABLE = False

# =====================================================
# CONFIG
# =====================================================
RNG = 42
np.random.seed(RNG)

RESULTS_DIR = "./results"
MODELS_DIR = "./models"
PROCESSED_PATH = os.path.join(RESULTS_DIR, "processed_ev_data.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---- CANONICAL COLUMN NAMES (LOCKED)
COL_START = "Charging Start Time"
COL_END = "Charging End Time"
COL_ENERGY = "Energy Consumed (kWh)"

# ---- Physics constants
ALPHA = 0.3
CHARGER_MAX_KW = 7.2
V2G_EXPORT_LIMIT = 3.6

# ---- FEATURE CONTRACT (DO NOT CHANGE)
SESSION_FEATURES = [
    "duration_min",
    COL_ENERGY,
    "start_hour",
    "day_of_week",
    "is_weekend",
    "soc_est"
]

TARGET_COL = "physics_flexible_kW"

# =====================================================
# FIX 1: HARD COLUMN NORMALIZATION (FINAL)
# =====================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce ONE canonical schema forever.
    """
    df = df.copy()

    # normalize column text
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

    rename_map = {
        "Energy Consumed kWh": COL_ENERGY,
        "Energy Consumed (kwh)": COL_ENERGY,
        "Energy Consumed(kWh)": COL_ENERGY,
        "Energy Consumed Kwh": COL_ENERGY,
        "Charging Start Time": COL_START,
        "Charging End Time": COL_END,
        "Day of Week": "day_of_week",
        "Start Hour": "start_hour"
    }

    df.rename(columns=rename_map, inplace=True)

    # drop duplicate columns created by renaming
    df = df.loc[:, ~df.columns.duplicated()]

    return df

# =====================================================
# PREPROCESSING
# =====================================================
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    df = normalize_columns(df)
    df = df.drop_duplicates()

    for c in [COL_START, COL_END, COL_ENERGY]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    df[COL_START] = pd.to_datetime(df[COL_START], errors="coerce")
    df[COL_END] = pd.to_datetime(df[COL_END], errors="coerce")
    df[COL_ENERGY] = pd.to_numeric(df[COL_ENERGY], errors="coerce")

    df = df.dropna(subset=[COL_START, COL_END, COL_ENERGY])

    # ---- Time features
    df["duration_min"] = (df[COL_END] - df[COL_START]).dt.total_seconds() / 60
    df = df[df["duration_min"] > 0]
    df["duration_hr"] = df["duration_min"] / 60

    df["start_hour"] = df[COL_START].dt.hour
    df["day_of_week"] = df[COL_START].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ---- SoC proxy
    df["soc_est"] = np.clip(
        0.3 + 0.7 * (df[COL_ENERGY] / df[COL_ENERGY].max()),
        0.2, 0.95
    )

    # ---- Physics target
    df["avg_charging_kW"] = df[COL_ENERGY] / df["duration_hr"]
    df["physical_max_kW"] = df["avg_charging_kW"].clip(upper=CHARGER_MAX_KW)

    df[TARGET_COL] = (
        df["physical_max_kW"] * ALPHA * df["soc_est"]
    ).clip(upper=V2G_EXPORT_LIMIT)

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"✅ Processed dataset saved → {PROCESSED_PATH}")

    return df

# =====================================================
# FIX 2: LOAD OR PREPARE (SELF-HEALING)
# =====================================================
def load_or_prepare() -> pd.DataFrame:

    if os.path.exists(PROCESSED_PATH):
        print("[INFO] Loading processed dataset")
        return preprocess_dataframe(pd.read_csv(PROCESSED_PATH))

    if not KAGGLE_AVAILABLE:
        raise RuntimeError(
            "Processed dataset not found and kagglehub unavailable.\n"
            "Either place processed_ev_data.csv in ./results/\n"
            "or install kagglehub."
        )

    print("[INFO] Processed dataset not found — downloading raw data")
    path = dataset_download("valakhorasani/electric-vehicle-charging-patterns")
    csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
    raw_df = pd.read_csv(os.path.join(path, csv_file))

    return preprocess_dataframe(raw_df)

# =====================================================
# SESSION MODELS
# =====================================================
def train_session_models(df: pd.DataFrame):

    df = df.dropna(subset=SESSION_FEATURES + [TARGET_COL]).reset_index(drop=True)

    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train, y_train = train[SESSION_FEATURES], train[TARGET_COL]
    X_test, y_test = test[SESSION_FEATURES], test[TARGET_COL]

    point_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=RNG
    )
    point_model.fit(X_train, y_train)

    joblib.dump(point_model, f"{MODELS_DIR}/lightgbm_point_model.pkl")

    rmse = np.sqrt(mean_squared_error(y_test, point_model.predict(X_test)))
    print(f"📌 Session Point RMSE: {rmse:.3f}")

    for q in [0.1, 0.5, 0.9]:
        gbr = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=RNG
        )
        gbr.fit(X_train, y_train)
        joblib.dump(gbr, f"{MODELS_DIR}/quantile_q{int(q*100)}.pkl")

    print("✅ Session models trained")

# =====================================================
# TIME-SERIES MODEL
# =====================================================
def train_timeseries_model(df: pd.DataFrame):

    print("\n⏳ Training time-series model")

    df_ts = df.copy()
    df_ts["timestamp"] = df_ts[COL_START].dt.floor("h")

    hourly = (
        df_ts.groupby("timestamp")
        .agg({
            TARGET_COL: "sum",
            "start_hour": "first",
            "day_of_week": "first",
            "is_weekend": "first"
        })
        .reset_index()
    )

    for lag in [1, 2, 3]:
        hourly[f"lag_{lag}"] = hourly[TARGET_COL].shift(lag)

    hourly = hourly.dropna().reset_index(drop=True)

    TS_FEATURES = [
        "start_hour",
        "day_of_week",
        "is_weekend",
        "lag_1",
        "lag_2",
        "lag_3"
    ]

    X = hourly[TS_FEATURES]
    y = hourly[TARGET_COL]

    split = int(len(hourly) * 0.8)
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        random_state=RNG
    )
    model.fit(X[:split], y[:split])

    rmse = np.sqrt(mean_squared_error(y[split:], model.predict(X[split:])))
    print(f"📈 Time-series RMSE: {rmse:.3f}")

    joblib.dump(model, f"{MODELS_DIR}/lightgbm_timeseries_model.pkl")
    print("✅ Time-series model saved")

# =====================================================
# MAIN
# =====================================================
def main():
    df = load_or_prepare()
    train_session_models(df)
    train_timeseries_model(df)
    print("\n🎉 ALL MODELS TRAINED WITH LOCKED SCHEMA")

if __name__ == "__main__":
    main()
