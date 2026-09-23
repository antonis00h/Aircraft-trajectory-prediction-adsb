# -*- coding: utf-8 -*-
"""
Prediction of Aircraft Route/Location
MSc thesis code

This script reproduces the main experimental pipeline used in the thesis:
1. Baseline delta-feature LSTM experiments (5, 10, 15 min)
2. Enriched-feature LSTM experiments
3. Architecture comparison
4. Final ENRICH_LSTM96 training for 10- and 15-minute horizons
5. Full-day robustness evaluation
6. Derived ETA-style temporal analysis

Designed for Google Colab with data stored in Google Drive.
"""

# =============================================================================
# 1. Imports and configuration
# =============================================================================

from google.colab import drive
drive.mount("/content/drive")

import glob
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models


BASE_DIR = "/content/drive/MyDrive"
DATA_DIR = os.path.join(BASE_DIR, "exports_hourly")
MODEL_DIR = os.path.join(BASE_DIR, "opensky_models")
RESULTS_DIR = os.path.join(BASE_DIR, "opensky_results")

DEV_DATE = "20190501"
EVAL_DATE = "20190502"

INPUT_LEN = 120
MAX_SAMPLES = 200_000
SEGMENT_DT = 60
MIN_LEN = 300
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# 2. Data loading and preprocessing helpers
# =============================================================================

COLS = [
    "icao24",
    "callsign",
    "time",
    "lat",
    "lon",
    "baroaltitude",
    "velocity",
    "heading",
    "vertrate",
]

DTYPES = {
    "icao24": "string",
    "callsign": "string",
    "time": "int64",
    "lat": "float32",
    "lon": "float32",
    "baroaltitude": "float32",
    "velocity": "float32",
    "heading": "float32",
    "vertrate": "float32",
}


def read_opensky_csv(path):
    """Read an hourly OpenSky CSV export used in this thesis."""
    return pd.read_csv(
        path,
        encoding="utf-16",
        header=None,
        names=COLS,
        dtype=DTYPES,
        sep=",",
    )


def preprocess_base(df, segment_dt=SEGMENT_DT, min_len=MIN_LEN):
    """Clean, order, segment, and create positional delta features."""
    df = df.copy()

    if "callsign" in df.columns:
        df["callsign"] = df["callsign"].astype("string").str.strip()

    df = df.dropna(
        subset=["icao24", "time", "lat", "lon", "baroaltitude"]
    ).copy()

    df = df.sort_values(["icao24", "time"]).reset_index(drop=True)

    df["dt"] = df.groupby("icao24")["time"].diff()
    df["new_segment"] = (df["dt"] > segment_dt) | (df["dt"].isna())
    df["segment_id"] = df.groupby("icao24")["new_segment"].cumsum()
    df["traj_id"] = df["icao24"] + "_" + df["segment_id"].astype(str)

    lengths = df.groupby("traj_id").size()
    valid_trajs = lengths[lengths >= min_len].index
    df = df[df["traj_id"].isin(valid_trajs)].reset_index(drop=True)

    df["dlat"] = df.groupby("traj_id")["lat"].diff()
    df["dlon"] = df.groupby("traj_id")["lon"].diff()
    df["dalt"] = df.groupby("traj_id")["baroaltitude"].diff()

    df = df.dropna(subset=["dlat", "dlon", "dalt"]).reset_index(drop=True)
    return df


def preprocess_enriched(df, segment_dt=SEGMENT_DT, min_len=MIN_LEN):
    """Apply base preprocessing and create the seven enriched input features."""
    df = df.dropna(
        subset=[
            "icao24",
            "time",
            "lat",
            "lon",
            "baroaltitude",
            "velocity",
            "heading",
            "vertrate",
        ]
    ).copy()

    df = preprocess_base(df, segment_dt=segment_dt, min_len=min_len)

    hdg_rad = np.deg2rad(df["heading"].to_numpy(dtype=np.float32))
    df["sin_hdg"] = np.sin(hdg_rad).astype(np.float32)
    df["cos_hdg"] = np.cos(hdg_rad).astype(np.float32)

    df["velocity"] = df["velocity"].clip(lower=0, upper=400).astype(np.float32)
    df["vertrate"] = df["vertrate"].clip(lower=-50, upper=50).astype(np.float32)

    return df


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometers."""
    radius_km = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return radius_km * c


# =============================================================================
# 3. Time-based dataset construction
# =============================================================================

def build_delta_dataset_timebased(
    df,
    input_len=INPUT_LEN,
    H=300,
    max_samples=MAX_SAMPLES,
    per_traj_cap=200,
    seed=SEED,
    tolerance_sec=10,
):
    """Create the 3-feature baseline sequence-to-one displacement dataset."""
    X = np.zeros((max_samples, input_len, 3), dtype=np.float32)
    y = np.zeros((max_samples, 3), dtype=np.float32)
    curr = np.zeros((max_samples, 3), dtype=np.float32)

    filled = 0
    rng = np.random.default_rng(seed)

    for _, g in df.groupby("traj_id"):
        g = g.sort_values("time").reset_index(drop=True)

        feats = g[["dlat", "dlon", "dalt"]].to_numpy(dtype=np.float32)
        abspos = g[["lat", "lon", "baroaltitude"]].to_numpy(dtype=np.float32)
        times = g["time"].to_numpy()

        length = len(g)
        if length <= input_len + 1:
            continue

        candidate_t = np.arange(input_len, length - 1)
        if len(candidate_t) == 0:
            continue

        per_traj = min(per_traj_cap, len(candidate_t))
        t_choices = rng.choice(candidate_t, size=per_traj, replace=False)

        for t in t_choices:
            if filled >= max_samples:
                break

            target_time = times[t] + H
            future_idx = np.searchsorted(times, target_time, side="left")

            if future_idx >= length:
                continue

            actual_gap = times[future_idx] - times[t]
            if abs(actual_gap - H) > tolerance_sec:
                continue

            X[filled] = feats[t - input_len:t]
            y[filled] = abspos[future_idx] - abspos[t]
            curr[filled] = abspos[t]
            filled += 1

        if filled >= max_samples:
            break

    return X[:filled], y[:filled], curr[:filled]


def build_delta_dataset_enriched_timebased(
    df,
    input_len=INPUT_LEN,
    H=300,
    max_samples=MAX_SAMPLES,
    per_traj_cap=200,
    seed=SEED,
    tolerance_sec=10,
):
    """Create the 7-feature enriched sequence-to-one displacement dataset."""
    X = np.zeros((max_samples, input_len, 7), dtype=np.float32)
    y = np.zeros((max_samples, 3), dtype=np.float32)
    curr = np.zeros((max_samples, 3), dtype=np.float32)

    filled = 0
    rng = np.random.default_rng(seed)

    feature_cols = [
        "dlat",
        "dlon",
        "dalt",
        "velocity",
        "sin_hdg",
        "cos_hdg",
        "vertrate",
    ]

    for _, g in df.groupby("traj_id"):
        g = g.sort_values("time").reset_index(drop=True)

        feats = g[feature_cols].to_numpy(dtype=np.float32)
        abspos = g[["lat", "lon", "baroaltitude"]].to_numpy(dtype=np.float32)
        times = g["time"].to_numpy()

        length = len(g)
        if length <= input_len + 1:
            continue

        candidate_t = np.arange(input_len, length - 1)
        if len(candidate_t) == 0:
            continue

        per_traj = min(per_traj_cap, len(candidate_t))
        t_choices = rng.choice(candidate_t, size=per_traj, replace=False)

        for t in t_choices:
            if filled >= max_samples:
                break

            target_time = times[t] + H
            future_idx = np.searchsorted(times, target_time, side="left")

            if future_idx >= length:
                continue

            actual_gap = times[future_idx] - times[t]
            if abs(actual_gap - H) > tolerance_sec:
                continue

            X[filled] = feats[t - input_len:t]
            y[filled] = abspos[future_idx] - abspos[t]
            curr[filled] = abspos[t]
            filled += 1

        if filled >= max_samples:
            break

    return X[:filled], y[:filled], curr[:filled]


# =============================================================================
# 4. Training and evaluation helpers
# =============================================================================

def train_eval_lstm_delta(
    X,
    y,
    curr,
    epochs=20,
    batch_size=256,
    seed=SEED,
):
    """Train and evaluate the baseline 64-unit LSTM."""
    n_samples = X.shape[0]

    idx = np.arange(n_samples)
    np.random.seed(seed)
    np.random.shuffle(idx)

    X = X[idx]
    y = y[idx]
    curr = curr[idx]

    train_end = int(0.80 * n_samples)
    val_end = int(0.90 * n_samples)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    curr_test = curr[val_end:]

    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)
    X_val_s = x_scaler.transform(
        X_val.reshape(-1, X_val.shape[-1])
    ).reshape(X_val.shape)
    X_test_s = x_scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)

    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s = y_scaler.transform(y_val)

    tf.random.set_seed(seed)

    seq_in = layers.Input(shape=(X.shape[1], X.shape[2]))
    x = layers.LSTM(64, return_sequences=False)(seq_in)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(3)(x)

    model = models.Model(inputs=seq_in, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train_s,
        y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    y_pred_s = model.predict(X_test_s, batch_size=512, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_s)

    abs_pred = curr_test + y_pred
    abs_true = curr_test + y_test

    dist_km = haversine_km(
        abs_true[:, 0],
        abs_true[:, 1],
        abs_pred[:, 0],
        abs_pred[:, 1],
    )
    alt_err = np.abs(abs_true[:, 2] - abs_pred[:, 2])

    return {
        "N_samples": int(n_samples),
        "epochs_ran": len(history.history["loss"]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "km_mean": float(dist_km.mean()),
        "km_median": float(np.median(dist_km)),
        "km_p90": float(np.percentile(dist_km, 90)),
        "alt_mean_m": float(alt_err.mean()),
        "alt_median_m": float(np.median(alt_err)),
    }


def train_eval_lstm_delta_config(
    X,
    y,
    curr,
    config,
    epochs=20,
    batch_size=256,
    seed=SEED,
):
    """Train and evaluate one architecture from the enriched LSTM family."""
    n_samples = X.shape[0]

    idx = np.arange(n_samples)
    np.random.seed(seed)
    np.random.shuffle(idx)

    X = X[idx]
    y = y[idx]
    curr = curr[idx]

    train_end = int(0.80 * n_samples)
    val_end = int(0.90 * n_samples)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    curr_test = curr[val_end:]

    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)
    X_val_s = x_scaler.transform(
        X_val.reshape(-1, X_val.shape[-1])
    ).reshape(X_val.shape)
    X_test_s = x_scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)

    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s = y_scaler.transform(y_val)

    tf.random.set_seed(seed)

    seq_in = layers.Input(shape=(X.shape[1], X.shape[2]))

    if config["type"] == "single":
        x = layers.LSTM(
            config["units"],
            return_sequences=False,
        )(seq_in)

        if config.get("dropout", 0.0) > 0:
            x = layers.Dropout(config["dropout"])(x)

    elif config["type"] == "stacked":
        x = layers.LSTM(
            config["units1"],
            return_sequences=True,
        )(seq_in)

        if config.get("dropout", 0.0) > 0:
            x = layers.Dropout(config["dropout"])(x)

        x = layers.LSTM(
            config["units2"],
            return_sequences=False,
        )(x)

        if config.get("dropout", 0.0) > 0:
            x = layers.Dropout(config["dropout"])(x)

    else:
        raise ValueError("Unknown config type")

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(3)(x)

    model = models.Model(inputs=seq_in, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train_s,
        y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    y_pred_s = model.predict(X_test_s, batch_size=512, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_s)

    abs_pred = curr_test + y_pred
    abs_true = curr_test + y_test

    dist_km = haversine_km(
        abs_true[:, 0],
        abs_true[:, 1],
        abs_pred[:, 0],
        abs_pred[:, 1],
    )
    alt_err = np.abs(abs_true[:, 2] - abs_pred[:, 2])

    return {
        "model": config["name"],
        "N_samples": int(n_samples),
        "epochs_ran": len(history.history["loss"]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "km_mean": float(dist_km.mean()),
        "km_median": float(np.median(dist_km)),
        "km_p90": float(np.percentile(dist_km, 90)),
        "alt_mean_m": float(alt_err.mean()),
        "alt_median_m": float(np.median(alt_err)),
    }


# =============================================================================
# 5. Development data
# =============================================================================

dev_file = os.path.join(
    DATA_DIR,
    f"opensky_FL200_{DEV_DATE}_00.csv",
)

if not os.path.exists(dev_file):
    raise FileNotFoundError(
        f"Development file not found: {dev_file}"
    )

df_dev_raw = read_opensky_csv(dev_file)
df_base = preprocess_base(df_dev_raw)

print("Development rows after base preprocessing:", len(df_base))
print("Trajectory segments:", df_base["traj_id"].nunique())


# =============================================================================
# 6. Baseline experiments: 3 delta features
# =============================================================================

BASE_HORIZONS = [
    ("5min", 300),
    ("10min", 600),
    ("15min", 900),
]

baseline_results = []

for label, horizon_sec in BASE_HORIZONS:
    print(f"\nBaseline horizon: {label}")

    X, y, curr = build_delta_dataset_timebased(
        df_base,
        input_len=INPUT_LEN,
        H=horizon_sec,
        max_samples=MAX_SAMPLES,
        seed=SEED,
    )

    metrics = train_eval_lstm_delta(
        X,
        y,
        curr,
        epochs=20,
        batch_size=256,
        seed=SEED,
    )
    metrics["label"] = label
    metrics["H_sec"] = horizon_sec
    baseline_results.append(metrics)

baseline_df = pd.DataFrame(baseline_results).sort_values("H_sec")
print(baseline_df[
    [
        "label",
        "H_sec",
        "N_samples",
        "epochs_ran",
        "km_mean",
        "km_median",
        "km_p90",
        "alt_mean_m",
        "alt_median_m",
    ]
])

plt.figure()
plt.plot(
    baseline_df["H_sec"] / 60,
    baseline_df["km_mean"],
    marker="o",
    label="Mean km",
)
plt.plot(
    baseline_df["H_sec"] / 60,
    baseline_df["km_median"],
    marker="o",
    label="Median km",
)
plt.plot(
    baseline_df["H_sec"] / 60,
    baseline_df["km_p90"],
    marker="o",
    label="P90 km",
)
plt.xlabel("Horizon (minutes)")
plt.ylabel("Haversine error (km)")
plt.title("Baseline trajectory prediction error vs horizon")
plt.legend()
plt.show()


# =============================================================================
# 7. Feature enrichment experiments
# =============================================================================

# Preserve the development-stage sequence used in the original thesis code:
# start from the already segmented base-development dataframe, then add the
# enriched kinematic features without rebuilding the trajectory segmentation.
df_enriched = df_base.dropna(
    subset=["velocity", "heading", "vertrate"]
).reset_index(drop=True)

hdg_rad = np.deg2rad(df_enriched["heading"].to_numpy(dtype=np.float32))
df_enriched["sin_hdg"] = np.sin(hdg_rad).astype(np.float32)
df_enriched["cos_hdg"] = np.cos(hdg_rad).astype(np.float32)
df_enriched["velocity"] = (
    df_enriched["velocity"].clip(lower=0, upper=400).astype(np.float32)
)
df_enriched["vertrate"] = (
    df_enriched["vertrate"].clip(lower=-50, upper=50).astype(np.float32)
)

enriched_results = []

for label, horizon_sec in BASE_HORIZONS:
    print(f"\nEnriched horizon: {label}")

    X, y, curr = build_delta_dataset_enriched_timebased(
        df_enriched,
        input_len=INPUT_LEN,
        H=horizon_sec,
        max_samples=MAX_SAMPLES,
        seed=SEED,
    )

    metrics = train_eval_lstm_delta(
        X,
        y,
        curr,
        epochs=20,
        batch_size=256,
        seed=SEED,
    )
    metrics["label"] = label
    metrics["H_sec"] = horizon_sec
    enriched_results.append(metrics)

enriched_df = pd.DataFrame(enriched_results).sort_values("H_sec")
print(enriched_df[
    [
        "label",
        "H_sec",
        "N_samples",
        "epochs_ran",
        "km_mean",
        "km_median",
        "km_p90",
        "alt_mean_m",
        "alt_median_m",
    ]
])

plt.figure()
plt.plot(
    baseline_df["H_sec"] / 60,
    baseline_df["km_mean"],
    marker="o",
    label="Baseline mean",
)
plt.plot(
    enriched_df["H_sec"] / 60,
    enriched_df["km_mean"],
    marker="o",
    label="Enriched mean",
)
plt.xlabel("Horizon (minutes)")
plt.ylabel("Haversine error (km)")
plt.title("Mean error: Baseline vs enriched features")
plt.legend()
plt.show()

plt.figure()
plt.plot(
    baseline_df["H_sec"] / 60,
    baseline_df["km_p90"],
    marker="o",
    label="Baseline p90",
)
plt.plot(
    enriched_df["H_sec"] / 60,
    enriched_df["km_p90"],
    marker="o",
    label="Enriched p90",
)
plt.xlabel("Horizon (minutes)")
plt.ylabel("Haversine error p90 (km)")
plt.title("P90 error: Baseline vs enriched features")
plt.legend()
plt.show()


# =============================================================================
# 8. Architecture comparison
# =============================================================================

ARCH_HORIZONS = [
    ("10min", 600),
    ("15min", 900),
]

CONFIGS = [
    {
        "name": "ENRICH_LSTM64",
        "type": "single",
        "units": 64,
        "dropout": 0.0,
    },
    {
        "name": "ENRICH_LSTM96",
        "type": "single",
        "units": 96,
        "dropout": 0.0,
    },
    {
        "name": "ENRICH_LSTM64x32_DO20",
        "type": "stacked",
        "units1": 64,
        "units2": 32,
        "dropout": 0.2,
    },
]

architecture_results = []

for label, horizon_sec in ARCH_HORIZONS:
    print(f"\nArchitecture comparison dataset: {label}")

    X, y, curr = build_delta_dataset_enriched_timebased(
        df_enriched,
        input_len=INPUT_LEN,
        H=horizon_sec,
        max_samples=MAX_SAMPLES,
        seed=SEED,
    )

    for config in CONFIGS:
        print(f"Training {config['name']} on {label}")

        metrics = train_eval_lstm_delta_config(
            X,
            y,
            curr,
            config=config,
            epochs=20,
            batch_size=256,
            seed=SEED,
        )
        metrics["label"] = label
        metrics["H_sec"] = horizon_sec
        architecture_results.append(metrics)

architecture_df = pd.DataFrame(architecture_results).sort_values(
    ["H_sec", "model"]
)

print(architecture_df[
    [
        "label",
        "model",
        "N_samples",
        "epochs_ran",
        "km_mean",
        "km_median",
        "km_p90",
        "alt_mean_m",
        "alt_median_m",
    ]
])

architecture_results_path = os.path.join(
    RESULTS_DIR,
    "architecture_comparison_results.csv",
)
architecture_df.to_csv(architecture_results_path, index=False)
print("Saved:", architecture_results_path)

architecture_df["H_min"] = architecture_df["H_sec"] / 60

for metric, title, ylabel in [
    (
        "km_mean",
        "Mean Haversine error vs horizon (enriched models)",
        "Mean error (km)",
    ),
    (
        "km_p90",
        "P90 Haversine error vs horizon (enriched models)",
        "P90 error (km)",
    ),
]:
    plt.figure()
    for model_name in architecture_df["model"].unique():
        subset = architecture_df[
            architecture_df["model"] == model_name
        ].sort_values("H_min")
        plt.plot(
            subset["H_min"],
            subset[metric],
            marker="o",
            label=model_name,
        )
    plt.xlabel("Horizon (minutes)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

pivot = architecture_df.pivot(
    index="model",
    columns="H_min",
    values="km_p90",
).sort_index()

model_names = pivot.index.tolist()
horizon_minutes = sorted(pivot.columns.tolist())
x_pos = np.arange(len(model_names))
width = 0.35

plt.figure()
plt.bar(
    x_pos - width / 2,
    pivot[horizon_minutes[0]].values,
    width,
    label=f"{int(horizon_minutes[0])} min",
)
plt.bar(
    x_pos + width / 2,
    pivot[horizon_minutes[1]].values,
    width,
    label=f"{int(horizon_minutes[1])} min",
)
plt.xticks(x_pos, model_names, rotation=30, ha="right")
plt.ylabel("P90 error (km)")
plt.title("P90 comparison across model variants")
plt.legend()
plt.show()


# =============================================================================
# 9. Final ENRICH_LSTM96 training on two development files
# =============================================================================

final_training_files = [
    os.path.join(DATA_DIR, f"opensky_FL200_{DEV_DATE}_00.csv"),
    os.path.join(DATA_DIR, f"opensky_FL200_{DEV_DATE}_01.csv"),
]

missing_training_files = [
    path for path in final_training_files if not os.path.exists(path)
]
if missing_training_files:
    raise FileNotFoundError(
        "Missing final-training files:\n" + "\n".join(missing_training_files)
    )

df_final_raw = pd.concat(
    [read_opensky_csv(path) for path in final_training_files],
    ignore_index=True,
)
df_final = preprocess_enriched(df_final_raw)

FINAL_HORIZONS = [
    ("10min", 600),
    ("15min", 900),
]

for label, horizon_sec in FINAL_HORIZONS:
    print(f"\nTraining final ENRICH_LSTM96: {label}")

    X, y, _ = build_delta_dataset_enriched_timebased(
        df_final,
        input_len=INPUT_LEN,
        H=horizon_sec,
        max_samples=MAX_SAMPLES,
        seed=SEED,
    )

    n_samples = X.shape[0]
    idx = np.arange(n_samples)
    np.random.seed(SEED)
    np.random.shuffle(idx)

    X = X[idx]
    y = y[idx]

    train_end = int(0.80 * n_samples)
    val_end = int(0.90 * n_samples)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]

    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)
    X_val_s = x_scaler.transform(
        X_val.reshape(-1, X_val.shape[-1])
    ).reshape(X_val.shape)

    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s = y_scaler.transform(y_val)

    tf.random.set_seed(SEED)

    seq_in = layers.Input(shape=(X.shape[1], X.shape[2]))
    x = layers.LSTM(96, return_sequences=False)(seq_in)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(3)(x)

    model = models.Model(seq_in, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    model.fit(
        X_train_s,
        y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=20,
        batch_size=256,
        callbacks=callbacks,
        verbose=1,
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"ENRICH_LSTM96_{label}.keras",
    )
    x_scaler_path = os.path.join(
        MODEL_DIR,
        f"ENRICH_LSTM96_{label}_x_scaler.pkl",
    )
    y_scaler_path = os.path.join(
        MODEL_DIR,
        f"ENRICH_LSTM96_{label}_y_scaler.pkl",
    )

    model.save(model_path)
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)

    print("Saved:", model_path)
    print("Saved:", x_scaler_path)
    print("Saved:", y_scaler_path)


# =============================================================================
# 10. Load final models and scalers
# =============================================================================

loaded = {}

for label in ["10min", "15min"]:
    loaded[label] = {
        "model": tf.keras.models.load_model(
            os.path.join(
                MODEL_DIR,
                f"ENRICH_LSTM96_{label}.keras",
            )
        ),
        "x_scaler": joblib.load(
            os.path.join(
                MODEL_DIR,
                f"ENRICH_LSTM96_{label}_x_scaler.pkl",
            )
        ),
        "y_scaler": joblib.load(
            os.path.join(
                MODEL_DIR,
                f"ENRICH_LSTM96_{label}_y_scaler.pkl",
            )
        ),
    }

print("Loaded final models and scalers.")


# =============================================================================
# 11. Full-day spatial robustness evaluation
# =============================================================================

def eval_model_on_dataset_scaled(
    model,
    X,
    y,
    curr,
    x_scaler,
    y_scaler,
    batch_size=512,
):
    X_s = x_scaler.transform(
        X.reshape(-1, X.shape[-1])
    ).reshape(X.shape)

    y_pred_s = model.predict(
        X_s,
        batch_size=batch_size,
        verbose=0,
    )
    y_pred = y_scaler.inverse_transform(y_pred_s)

    abs_pred = curr + y_pred
    abs_true = curr + y

    dist_km = haversine_km(
        abs_true[:, 0],
        abs_true[:, 1],
        abs_pred[:, 0],
        abs_pred[:, 1],
    )
    alt_err = np.abs(abs_true[:, 2] - abs_pred[:, 2])

    return {
        "N_samples": int(X.shape[0]),
        "km_mean": float(dist_km.mean()),
        "km_median": float(np.median(dist_km)),
        "km_p90": float(np.percentile(dist_km, 90)),
        "alt_mean_m": float(alt_err.mean()),
        "alt_median_m": float(np.median(alt_err)),
    }


full_day_results_path = os.path.join(
    RESULTS_DIR,
    f"full_day_{EVAL_DATE}_ENRICH_LSTM96.csv",
)

full_day_rows = []

for hour in range(24):
    hh = f"{hour:02d}"
    filename = f"opensky_FL200_{EVAL_DATE}_{hh}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        print("Missing file:", filename)
        continue

    print(f"\nFull-day evaluation hour: {hh}")

    df_hour = preprocess_enriched(
        read_opensky_csv(filepath),
        segment_dt=SEGMENT_DT,
        min_len=MIN_LEN,
    )

    for label, horizon_sec in FINAL_HORIZONS:
        X, y, curr = build_delta_dataset_enriched_timebased(
            df_hour,
            input_len=INPUT_LEN,
            H=horizon_sec,
            max_samples=MAX_SAMPLES,
            seed=SEED,
        )

        if X.shape[0] == 0:
            print(label, ": no samples")
            continue

        metrics = eval_model_on_dataset_scaled(
            loaded[label]["model"],
            X,
            y,
            curr,
            loaded[label]["x_scaler"],
            loaded[label]["y_scaler"],
        )

        full_day_rows.append(
            {
                "date": EVAL_DATE,
                "hour": hour,
                "horizon": label,
                "H_sec": horizon_sec,
                **metrics,
            }
        )

    pd.DataFrame(full_day_rows).to_csv(
        full_day_results_path,
        index=False,
    )

print("Saved:", full_day_results_path)

full_day_res = pd.read_csv(full_day_results_path)

full_day_summary = full_day_res.groupby("horizon").agg(
    hours=("hour", "nunique"),
    N_samples=("N_samples", "sum"),
    km_mean=("km_mean", "mean"),
    km_median=("km_median", "mean"),
    km_p90=("km_p90", "mean"),
    alt_mean_m=("alt_mean_m", "mean"),
    alt_median_m=("alt_median_m", "mean"),
).reset_index()

print(full_day_summary)

for metric in ["km_mean", "km_p90"]:
    plt.figure()
    for horizon in full_day_res["horizon"].unique():
        subset = full_day_res[
            full_day_res["horizon"] == horizon
        ].sort_values("hour")
        plt.plot(
            subset["hour"],
            subset[metric],
            marker="o",
            label=horizon,
        )
    plt.xlabel("Hourly file index")
    plt.ylabel(metric)
    plt.title(f"{metric} across the full-day evaluation set")
    plt.legend()
    plt.show()


# =============================================================================
# 12. Derived ETA-style temporal analysis
# =============================================================================

def eval_model_with_eta_scaled(
    model,
    X,
    y,
    curr,
    H_sec,
    x_scaler,
    y_scaler,
    batch_size=512,
    eps_km=1e-3,
    clip_sec=3600,
):
    X_s = x_scaler.transform(
        X.reshape(-1, X.shape[-1])
    ).reshape(X.shape)

    y_pred_s = model.predict(
        X_s,
        batch_size=batch_size,
        verbose=0,
    )
    y_pred = y_scaler.inverse_transform(y_pred_s)

    abs_pred = curr + y_pred
    abs_true = curr + y

    dist_km = haversine_km(
        abs_true[:, 0],
        abs_true[:, 1],
        abs_pred[:, 0],
        abs_pred[:, 1],
    )
    alt_err = np.abs(abs_true[:, 2] - abs_pred[:, 2])

    d_true = haversine_km(
        curr[:, 0],
        curr[:, 1],
        abs_true[:, 0],
        abs_true[:, 1],
    )
    d_pred = haversine_km(
        curr[:, 0],
        curr[:, 1],
        abs_pred[:, 0],
        abs_pred[:, 1],
    )

    d_pred_safe = np.maximum(d_pred, eps_km)

    eta_hat = H_sec * (d_true / d_pred_safe)
    eta_err = eta_hat - H_sec
    eta_err = np.clip(eta_err, -clip_sec, clip_sec)
    eta_abs = np.abs(eta_err)

    return {
        "N_samples": int(X.shape[0]),
        "km_mean": float(dist_km.mean()),
        "km_median": float(np.median(dist_km)),
        "km_p90": float(np.percentile(dist_km, 90)),
        "alt_mean_m": float(alt_err.mean()),
        "alt_median_m": float(np.median(alt_err)),
        "eta_mean_s": float(eta_err.mean()),
        "eta_median_s": float(np.median(eta_err)),
        "eta_p90_abs_s": float(np.percentile(eta_abs, 90)),
        "eta_mean_abs_s": float(eta_abs.mean()),
        "eta_median_abs_s": float(np.median(eta_abs)),
    }


eta_results_path = os.path.join(
    RESULTS_DIR,
    f"full_day_{EVAL_DATE}_ENRICH_LSTM96_with_ETA.csv",
)

eta_rows = []

for hour in range(24):
    hh = f"{hour:02d}"
    filename = f"opensky_FL200_{EVAL_DATE}_{hh}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        print("Missing file:", filename)
        continue

    print(f"\nETA analysis hour: {hh}")

    df_hour = preprocess_enriched(
        read_opensky_csv(filepath),
        segment_dt=SEGMENT_DT,
        min_len=MIN_LEN,
    )

    for label, horizon_sec in FINAL_HORIZONS:
        X, y, curr = build_delta_dataset_enriched_timebased(
            df_hour,
            input_len=INPUT_LEN,
            H=horizon_sec,
            max_samples=MAX_SAMPLES,
            seed=SEED,
        )

        if X.shape[0] == 0:
            print(label, ": no samples")
            continue

        metrics = eval_model_with_eta_scaled(
            loaded[label]["model"],
            X,
            y,
            curr,
            H_sec=horizon_sec,
            x_scaler=loaded[label]["x_scaler"],
            y_scaler=loaded[label]["y_scaler"],
        )

        eta_rows.append(
            {
                "date": EVAL_DATE,
                "hour": hour,
                "horizon": label,
                "H_sec": horizon_sec,
                **metrics,
            }
        )

        print(
            label,
            "ETA mean absolute error (min):",
            round(metrics["eta_mean_abs_s"] / 60, 2),
            "P90 absolute error (min):",
            round(metrics["eta_p90_abs_s"] / 60, 2),
        )

    pd.DataFrame(eta_rows).to_csv(
        eta_results_path,
        index=False,
    )

print("Saved:", eta_results_path)

eta_res = pd.read_csv(eta_results_path)

eta_summary = eta_res.groupby("horizon").agg(
    hours=("hour", "nunique"),
    N_samples=("N_samples", "sum"),
    km_mean=("km_mean", "mean"),
    km_p90=("km_p90", "mean"),
    eta_mean_abs_min=(
        "eta_mean_abs_s",
        lambda s: float(s.mean() / 60),
    ),
    eta_median_abs_min=(
        "eta_median_abs_s",
        lambda s: float(s.mean() / 60),
    ),
    eta_p90_abs_min=(
        "eta_p90_abs_s",
        lambda s: float(s.mean() / 60),
    ),
).reset_index()

print(eta_summary)

for metric, title in [
    ("eta_mean_abs_s", "Mean absolute ETA-style error over the day"),
    ("eta_p90_abs_s", "P90 absolute ETA-style error over the day"),
]:
    plt.figure()
    for horizon in eta_res["horizon"].unique():
        subset = eta_res[
            eta_res["horizon"] == horizon
        ].sort_values("hour")
        plt.plot(
            subset["hour"],
            subset[metric] / 60,
            marker="o",
            label=horizon,
        )
    plt.xlabel("Hourly file index")
    plt.ylabel("Absolute temporal error (minutes)")
    plt.title(title)
    plt.legend()
    plt.show()


# =============================================================================
# 13. Horizon-normalized error analysis
# =============================================================================

analysis_df = eta_res.copy()
analysis_df["H_min"] = analysis_df["H_sec"] / 60.0
analysis_df["eta_mean_abs_min"] = (
    analysis_df["eta_mean_abs_s"] / 60.0
)
analysis_df["eta_p90_abs_min"] = (
    analysis_df["eta_p90_abs_s"] / 60.0
)

plt.figure()
for horizon in sorted(analysis_df["horizon"].unique()):
    subset = analysis_df[analysis_df["horizon"] == horizon]
    plt.scatter(
        subset["km_mean"],
        subset["eta_mean_abs_min"],
        label=horizon,
        alpha=0.8,
    )
plt.xlabel("Mean horizontal error (km)")
plt.ylabel("Mean absolute ETA-style error (minutes)")
plt.title("Temporal proxy error vs spatial error")
plt.legend()
plt.show()

analysis_df["km_per_min_horizon"] = (
    analysis_df["km_mean"] / analysis_df["H_min"]
)
analysis_df["eta_min_per_min_horizon"] = (
    analysis_df["eta_mean_abs_min"] / analysis_df["H_min"]
)

plt.figure()
for horizon in sorted(analysis_df["horizon"].unique()):
    subset = analysis_df[
        analysis_df["horizon"] == horizon
    ].sort_values("hour")
    plt.plot(
        subset["hour"],
        subset["km_per_min_horizon"],
        marker="o",
        label=horizon,
    )
plt.xlabel("Hourly file index")
plt.ylabel("km error per minute of prediction horizon")
plt.title("Horizon-normalized spatial error")
plt.legend()
plt.show()

plt.figure()
for horizon in sorted(analysis_df["horizon"].unique()):
    subset = analysis_df[
        analysis_df["horizon"] == horizon
    ].sort_values("hour")
    plt.plot(
        subset["hour"],
        subset["eta_min_per_min_horizon"],
        marker="o",
        label=horizon,
    )
plt.xlabel("Hourly file index")
plt.ylabel("Temporal error per minute of prediction horizon")
plt.title("Horizon-normalized ETA-style temporal error")
plt.legend()
plt.show()

normalized_summary = analysis_df.groupby("horizon").agg(
    km_mean=("km_mean", "mean"),
    eta_mean_abs_min=("eta_mean_abs_min", "mean"),
    km_per_min=("km_per_min_horizon", "mean"),
    eta_min_per_min=("eta_min_per_min_horizon", "mean"),
).reset_index()

print(normalized_summary)
