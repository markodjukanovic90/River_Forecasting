# ============================================================
# FINAL PIPELINE: XGBoost + TimeSeries GridSearch + NSE
# River Bosna – Monthly Streamflow Forecasting
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.linear_model import LinearRegression
from functools import reduce

# ============================================================
# METRICS
# ============================================================

def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

def kge(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0,1]
    beta = np.mean(y_pred) / np.mean(y_true)
    gamma = (np.std(y_pred)/np.mean(y_pred)) / (np.std(y_true)/np.mean(y_true))
    return 1 - np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)

nse_scorer = make_scorer(nse, greater_is_better=True)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_lag_features(df, lags_P=[1,2,3], lags_T=[1,2], lag_Q=1):
    df = df.copy()
    P_cols = [c for c in df.columns if c.startswith("P_")]
    T_cols = [c for c in df.columns if c.startswith("T_")]

    for lag in lags_P:
        for c in P_cols:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)

    for lag in lags_T:
        for c in T_cols:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)

    df["Q_lag1"] = df["Q_proticaj"].shift(lag_Q)
    return df

# ============================================================
# MAIN PIPELINE
# ============================================================

def run_final_xgboost_pipeline(df):

    # -----------------------------
    # Time handling
    # -----------------------------
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # -----------------------------
    # Lag features
    # -----------------------------
    df = add_lag_features(df)
    df = df.dropna()

    X = df.drop(columns=["Q_proticaj"])
    y = df["Q_proticaj"]

    # -----------------------------
    # Train / Test split
    # -----------------------------
    train_idx = df.index.year <= 2010
    test_idx  = df.index.year >= 2011

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test,  y_test  = X.loc[test_idx],  y.loc[test_idx]

    # -----------------------------
    # TimeSeries CV
    # -----------------------------
    tscv = TimeSeriesSplit(n_splits=5)

    # -----------------------------
    # Parameter grid
    # -----------------------------
    param_grid = {
        "n_estimators": [300, 600, 900],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.03, 0.05],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9]
    }

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    )

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=nse_scorer,
        cv=tscv,
        n_jobs=-1,
        verbose=2
    )

    print("\n>>> RUNNING GRID SEARCH (TimeSeriesSplit)")
    grid.fit(X_train, y_train)

    print("\nBEST CV NSE:", grid.best_score_)
    print("BEST PARAMETERS:")
    for k,v in grid.best_params_.items():
        print(f"  {k}: {v}")

    # -----------------------------
    # Final model
    # -----------------------------
    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)

    # -----------------------------
    # Linear scaling bias correction
    # -----------------------------
    scaler = LinearRegression()
    scaler.fit(best_model.predict(X_train).reshape(-1,1), y_train.values)
    a, b = scaler.coef_[0], scaler.intercept_
    y_pred_corr = a * y_pred + b

    print(f"\nLinear-scaling: a={a:.3f}, b={b:.3f}")

    # -----------------------------
    # Metrics
    # -----------------------------
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_corr)),
        "MAE" : mean_absolute_error(y_test, y_pred_corr),
        "NSE" : nse(y_test.values, y_pred_corr),
        "KGE" : kge(y_test.values, y_pred_corr)
    }

    print("\nFINAL TEST (2011–2020)")
    for k,v in metrics.items():
        print(f"{k}: {v:.3f}")

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(11,5))
    plt.plot(y_test.index, y_test.values, label="Observed", lw=2)
    plt.plot(y_test.index, y_pred_corr, label="XGB Forecast (bias-corrected)", lw=2)
    plt.legend()
    plt.grid(True)
    plt.title("Monthly Streamflow – River Bosna")
    plt.tight_layout()
    plt.show()

    return metrics, grid.best_params_

# ============================================================
# USAGE
# ============================================================

# 0) PRIPREMA PODATAKA:

from convert import convert_data
import time 
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from functools import reduce


# -- padavine --
padavine_doboj = convert_data("padavine-doboj.ods")
padavine_tuzla = convert_data("padavine-tuzla.ods")
padavine_zenica = convert_data("padavine-zenica.ods")
padavine_sarajevo = convert_data("padavine-sarajevo.ods")
padavine_bjelasnica = convert_data("padavine-bjelasnica.ods")

# -- temperatura -- 
temp_doboj = convert_data("temp-doboj.ods", prefix="T")
temp_tuzla = convert_data("temp-tuzla.ods", prefix="T")
temp_zenica = convert_data("temp-zenica.ods", prefix="T")
temp_sarajevo = convert_data("temp-sarajevo.ods", prefix="T")
temp_bjelasnica = convert_data("temp-bjelasnica.ods", prefix="T")

# -- proticaj.ods -- 

proticaj = convert_data("proticaj-proticaj.ods", prefix="Q")

# NAN VALUES TREATING  temp_zenica Climatological Monthly Mean Filling (BEST for 6-month gap)
temp_zenica['date'] = pd.to_datetime(temp_zenica['date'])
temp_zenica = temp_zenica.set_index('date')

temp_zenica['month'] = temp_zenica.index.month
monthly_mean = temp_zenica.groupby('month')['T_zenica'].mean()

mask = (temp_zenica.index >= '2017-07-01') & (temp_zenica.index <= '2017-12-01')
temp_zenica.loc[mask, 'T_zenica'] = temp_zenica.loc[mask].index.month.map(monthly_mean)

#merge data: 
dfs = [
    padavine_doboj,
    padavine_tuzla,
    padavine_zenica,
    padavine_sarajevo,
    padavine_bjelasnica,
    temp_doboj,
    temp_tuzla,
    temp_zenica,
    temp_sarajevo,
    temp_bjelasnica,
    proticaj
]

df_merged = reduce(
    lambda left, right: pd.merge(left, right, on="date", how="inner"),
    dfs
)
print(df_merged)
metrics, best_params = run_final_xgboost_pipeline(df_merged)
