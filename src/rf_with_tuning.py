# ============================================================
# Random Forest hydrological forecasting with
# GridSearchCV + NSE + TimeSeriesSplit
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.linear_model import LinearRegression
from functools import reduce

from convert import convert_data   # assumed to exist


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def nse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)


def kge(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    r = np.corrcoef(y_true, y_pred)[0, 1]
    beta = np.mean(y_pred) / np.mean(y_true)
    gamma = (np.std(y_pred) / np.mean(y_pred)) / (np.std(y_true) / np.mean(y_true))

    return 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)


nse_scorer = make_scorer(nse, greater_is_better=True)


# ------------------------------------------------------------
# Lagged features
# ------------------------------------------------------------
def add_lag_features(df, lags_P=[1,2,3], lags_T=[1,2], lag_Q=1):
    df = df.copy()

    P_cols = [c for c in df.columns if c.startswith("P_")]
    T_cols = [c for c in df.columns if c.startswith("T_")]

    for lag in lags_P:
        for col in P_cols:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    for lag in lags_T:
        for col in T_cols:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    if "Q_proticaj" in df.columns:
        df["Q_lag1"] = df["Q_proticaj"].shift(lag_Q)

    return df


# ------------------------------------------------------------
# Main RF forecast with Grid Search
# ------------------------------------------------------------
def run_rf_forecast_with_lags(df,
                              target_col="Q_proticaj",
                              max_train_year=2010,
                              test_start_year=2011):

    # ---- datetime index ----
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    df = df.sort_index()

    # ---- lag features ----
    df = add_lag_features(df)
    df = df.dropna()

    # ---- X / y ----
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ---- train / test split ----
    train_idx = df.index.year <= max_train_year
    test_idx  = df.index.year >= test_start_year

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test,  y_test  = X.loc[test_idx],  y.loc[test_idx]

    # --------------------------------------------------------
    # Grid Search with TimeSeriesSplit
    # --------------------------------------------------------
    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [500, 1000, 2000],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    tscv = TimeSeriesSplit(n_splits=5)

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=nse_scorer,
        cv=tscv,
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    print("\nBest CV NSE:", grid.best_score_)
    print("Best parameters:")
    for k, v in grid.best_params_.items():
        print(f"  {k}: {v}")

    # --------------------------------------------------------
    # Prediction (1-step ahead)
    # --------------------------------------------------------
    y_pred = model.predict(X_test)

    # ---- Linear scaling (bias correction) ----
    scaler = LinearRegression()
    scaler.fit(model.predict(X_train).reshape(-1, 1), y_train.values)

    a, b = scaler.coef_[0], scaler.intercept_
    print(f"\nLinear scaling: a={a:.3f}, b={b:.3f}")

    y_pred_corr = a * y_pred + b

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_corr)),
        "MAE":  mean_absolute_error(y_test, y_pred_corr),
        "NSE":  nse(y_test, y_pred_corr),
        "KGE":  kge(y_test, y_pred_corr)
    }

    print("\nFORECAST METRICS (test period)")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    # --------------------------------------------------------
    # Plot hydrograph
    # --------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test.values, label="Observed", marker="o")
    plt.plot(y_test.index, y_pred, label="RF raw", marker="x")
    plt.plot(y_test.index, y_pred_corr, label="RF + Linear scaling", marker="s")
    plt.title("Observed vs Predicted Monthly Streamflow of BRB")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gs_rf_forecast_test_period.png", dpi=300)

    # --------------------------------------------------------
    # Feature importance with direction
    # --------------------------------------------------------
    import seaborn as sns

    # Feature importance magnitude from model
    imp = pd.Series(model.feature_importances_, index=X.columns)
    top_imp = imp.sort_values(ascending=False).head(5)  # top 10 features

    # Compute direction of influence (correlation with predictions)
    directions = []
    for f in top_imp.index:
        corr = np.corrcoef(X_test[f], y_pred_corr)[0, 1]
        directions.append(corr)

    # Put into DataFrame
    fi_df = pd.DataFrame({
        "feature": top_imp.index,
        "importance": top_imp.values,
        "direction": directions
    })  

    # Plot with color indicating direction
    plt.figure(figsize=(8,5))
    sns.barplot(
        x="importance", y="feature", data=fi_df,
        palette=["green" if d>0 else "red" for d in fi_df["direction"]]
    )
    plt.title("Top 5 RF Features: Importance & Direction")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("feature_importance_rf.png", dpi=300)
    plt.close()


    return y_pred_corr, metrics


# ============================================================
# DATA LOADING & MERGING (unchanged logic)
# ============================================================

padavine_doboj = convert_data("padavine-doboj.ods")
padavine_tuzla = convert_data("padavine-tuzla.ods")
padavine_zenica = convert_data("padavine-zenica.ods")
padavine_sarajevo = convert_data("padavine-sarajevo.ods")
padavine_bjelasnica = convert_data("padavine-bjelasnica.ods")

temp_doboj = convert_data("temp-doboj.ods", prefix="T")
temp_tuzla = convert_data("temp-tuzla.ods", prefix="T")
temp_zenica = convert_data("temp-zenica.ods", prefix="T")
temp_sarajevo = convert_data("temp-sarajevo.ods", prefix="T")
temp_bjelasnica = convert_data("temp-bjelasnica.ods", prefix="T")

proticaj = convert_data("proticaj-proticaj.ods", prefix="Q")

# ---- climatological filling ----
temp_zenica["date"] = pd.to_datetime(temp_zenica["date"])
temp_zenica = temp_zenica.set_index("date")

temp_zenica["month"] = temp_zenica.index.month
monthly_mean = temp_zenica.groupby("month")["T_zenica"].mean()

mask = (temp_zenica.index >= "2017-07-01") & (temp_zenica.index <= "2017-12-01")
temp_zenica.loc[mask, "T_zenica"] = temp_zenica.loc[mask].index.month.map(monthly_mean)

# ---- merge ----
dfs = [
    padavine_doboj, padavine_tuzla, padavine_zenica,
    padavine_sarajevo, padavine_bjelasnica,
    temp_doboj, temp_tuzla, temp_zenica,
    temp_sarajevo, temp_bjelasnica,
    proticaj
]

df_merged = reduce(lambda l, r: pd.merge(l, r, on="date", how="inner"), dfs)

# ============================================================
# RUN MODEL
# ============================================================
y_pred_corr, metrics = run_rf_forecast_with_lags(df_merged)

