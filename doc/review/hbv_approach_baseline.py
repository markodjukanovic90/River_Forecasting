# ============================================================
# HBV-LITE BASELINE FOR STREAMFLOW PREDICTION
# Consistent with ML pipeline (same data & split)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import product

from convert import convert_data


# ============================================================
# METRICS
# ============================================================
def nse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)


def kge(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0,1]
    beta = np.mean(y_pred) / np.mean(y_true)
    gamma = (np.std(y_pred)/np.mean(y_pred)) / (np.std(y_true)/np.mean(y_true))
    return 1 - np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)


# ============================================================
# HBV-LITE MODEL
# ============================================================
def hbv_lite(P, T, params):
    fc   = params["fc"]    # field capacity
    beta = params["beta"]  # nonlinearity
    k    = params["k"]     # runoff coefficient
    etf  = params["etf"]   # evapotranspiration factor

    SM = fc / 2  # initial soil moisture
    Q_sim = []

    for t in range(len(P)):
        # evapotranspiration
        ET = max(0, etf * T[t])

        # update soil moisture
        SM = SM + P[t] - ET
        SM = max(0, min(SM, fc))

        # recharge
        recharge = (SM / fc) ** beta * P[t]

        # runoff
        Q = k * recharge

        Q_sim.append(Q)

    return np.array(Q_sim)


# ============================================================
# PREPARE INPUTS (aggregate 5 stations)
# ============================================================
def prepare_hbv_inputs(df):
    P_cols = [c for c in df.columns if c.startswith("P_")]
    T_cols = [c for c in df.columns if c.startswith("T_")]

    P = df[P_cols].mean(axis=1).values
    T = df[T_cols].mean(axis=1).values
    Q = df["Q_proticaj"].values

    return P, T, Q


# ============================================================
# PARAMETER CALIBRATION (Grid Search with NSE)
# ============================================================
def calibrate_hbv(P, T, Q):
    param_grid = {
        "fc":   [100, 200, 300],
        "beta": [1, 2, 3],
        "k":    [0.1, 0.3, 0.5],
        "etf":  [0.1, 0.2, 0.3]
    }

    best_score = -np.inf
    best_params = None

    for combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))

        Q_sim = hbv_lite(P, T, params)
        score = nse(Q, Q_sim)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score


# ============================================================
# MAIN HBV PIPELINE
# ============================================================
def run_hbv_model(df):

    # --- datetime index ---
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # --- prepare inputs ---
    P, T, Q = prepare_hbv_inputs(df)

    # --- train/test split ---
    train_idx = df.index.year <= 2010
    test_idx  = df.index.year >= 2011

    P_train, T_train, Q_train = P[train_idx], T[train_idx], Q[train_idx]
    P_test,  T_test,  Q_test  = P[test_idx],  T[test_idx],  Q[test_idx]

    # --- calibration ---
    best_params, best_nse = calibrate_hbv(P_train, T_train, Q_train)

    print("\nBEST HBV PARAMETERS")
    print(best_params)
    print(f"Train NSE: {best_nse:.3f}")

    # --- simulation ---
    Q_train_sim = hbv_lite(P_train, T_train, best_params)
    Q_test_sim  = hbv_lite(P_test,  T_test,  best_params)

    # --- metrics ---
    metrics_train = {
        "RMSE": np.sqrt(mean_squared_error(Q_train, Q_train_sim)),
        "MAE":  mean_absolute_error(Q_train, Q_train_sim),
        "NSE":  nse(Q_train, Q_train_sim),
        "KGE":  kge(Q_train, Q_train_sim)
    }

    metrics_test = {
        "RMSE": np.sqrt(mean_squared_error(Q_test, Q_test_sim)),
        "MAE":  mean_absolute_error(Q_test, Q_test_sim),
        "NSE":  nse(Q_test, Q_test_sim),
        "KGE":  kge(Q_test, Q_test_sim)
    }

    print("\nHBV TRAIN METRICS")
    for k,v in metrics_train.items():
        print(f"{k}: {v:.3f}")

    print("\nHBV TEST METRICS")
    for k,v in metrics_test.items():
        print(f"{k}: {v:.3f}")

    # --- plot ---
    test_dates = df.index[test_idx]

    plt.figure(figsize=(10,5))
    plt.plot(test_dates, Q_test, label="Observed", marker='o')
    plt.plot(test_dates, Q_test_sim, label="HBV-lite", marker='s')
    plt.title("Observed vs HBV-lite Monthly Streamflow")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("hbv_forecast_test_period.png", dpi=300)
    plt.close()

    return metrics_test


# ============================================================
# DATA LOADING (same as your pipeline)
# ============================================================

# --- precipitation ---
padavine_doboj = convert_data("padavine-doboj.ods")
padavine_tuzla = convert_data("padavine-tuzla.ods")
padavine_zenica = convert_data("padavine-zenica.ods")
padavine_sarajevo = convert_data("padavine-sarajevo.ods")
padavine_bjelasnica = convert_data("padavine-bjelasnica.ods")

# --- temperature ---
temp_doboj = convert_data("temp-doboj.ods", prefix="T")
temp_tuzla = convert_data("temp-tuzla.ods", prefix="T")
temp_zenica = convert_data("temp-zenica.ods", prefix="T")
temp_sarajevo = convert_data("temp-sarajevo.ods", prefix="T")
temp_bjelasnica = convert_data("temp-bjelasnica.ods", prefix="T")

# --- discharge ---
proticaj = convert_data("proticaj-proticaj.ods", prefix="Q")


# --- missing data handling (Zenica 2017 gap) ---
temp_zenica["date"] = pd.to_datetime(temp_zenica["date"])
temp_zenica = temp_zenica.set_index("date")

temp_zenica["month"] = temp_zenica.index.month
monthly_mean = temp_zenica.groupby("month")["T_zenica"].mean()

mask = (temp_zenica.index >= "2017-07-01") & (temp_zenica.index <= "2017-12-01")
temp_zenica.loc[mask, "T_zenica"] = temp_zenica.loc[mask].index.month.map(monthly_mean)

temp_zenica = temp_zenica.reset_index()


# --- merge all ---
dfs = [
    padavine_doboj, padavine_tuzla, padavine_zenica,
    padavine_sarajevo, padavine_bjelasnica,
    temp_doboj, temp_tuzla, temp_zenica,
    temp_sarajevo, temp_bjelasnica,
    proticaj
]

df_merged = reduce(lambda l, r: pd.merge(l, r, on="date", how="inner"), dfs)

print(df_merged)

print("DF shape:", df_merged.shape)
print(df_merged.isna().sum().sum())
print(df_merged.head())
print(df_merged.tail())


# ============================================================
# RUN HBV MODEL
# ============================================================
metrics_hbv = run_hbv_model(df_merged)
print(metrics_hbv)
