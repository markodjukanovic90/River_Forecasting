import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from itertools import product
from convert import convert_data

# =====================================================
# Metrics
# =====================================================
def nse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

def kge(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0, 1]
    beta = np.mean(y_pred) / np.mean(y_true)
    gamma = (np.std(y_pred) / np.mean(y_pred)) / (np.std(y_true) / np.mean(y_true))
    return 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)

# =====================================================
# Lag features
# =====================================================
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

# =====================================================
# Build LSTM
# =====================================================
def build_lstm(n_features, units, lr):
    model = Sequential([
        LSTM(units, input_shape=(1, n_features)),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse"
    )
    return model

# =====================================================
# Time-series CV for NSE
# =====================================================
def time_series_cv_lstm(X, y, params, n_splits=5):
    fold_size = len(X) // (n_splits + 1)
    scores = []

    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        val_end = train_end + fold_size

        X_tr, X_val = X[:train_end], X[train_end:val_end]
        y_tr, y_val = y[:train_end], y[train_end:val_end]

        model = build_lstm(
            n_features=X.shape[2],
            units=params["units"],
            lr=params["lr"]
        )

        model.fit(
            X_tr, y_tr,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=0
        )

        y_val_pred = model.predict(X_val, verbose=0).flatten()
        scores.append(nse(y_val, y_val_pred))

    return np.mean(scores)

# =====================================================
# Grid Search (NSE-based)
# =====================================================
def grid_search_lstm(X_train, y_train):
    param_grid = {
        "units": [32, 64],
        "lr": [1e-3, 5e-4],
        "batch_size": [16, 32],
        "epochs": [50, 100, 200]
    }

    results = []

    for combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        print(f"Testing {params}")

        score = time_series_cv_lstm(X_train, y_train, params)
        params["NSE"] = score
        results.append(params)

        print(f" → Mean CV NSE = {score:.3f}")

    df_results = pd.DataFrame(results)
    return df_results.sort_values("NSE", ascending=False)

def run_lstm_with_nse_tuning(df, target_col="Q_proticaj"):
    import seaborn as sns
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Add lag features
    df = add_lag_features(df)
    df = df.dropna()

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    train_idx = df.index.year <= 2010
    test_idx  = df.index.year >= 2011

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Reshape for LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # ================= Grid Search =================
    gs_results = grid_search_lstm(X_train_lstm, y_train)
    best = gs_results.iloc[0].to_dict()

    print("\nBEST PARAMETERS")
    print(best)

    # ================= Train Best Model =================
    model = build_lstm(
        n_features=X_train_lstm.shape[2],
        units=int(best["units"]),
        lr=best["lr"]
    )

    model.fit(
        X_train_lstm, y_train,
        epochs=int(best["epochs"]),
        batch_size=int(best["batch_size"]),
        verbose=1
    )

    # ================= Prediction =================
    y_train_pred = model.predict(X_train_lstm).flatten()
    y_test_pred  = model.predict(X_test_lstm).flatten()

    # Linear scaling
    scaler = LinearRegression()
    scaler.fit(y_train_pred.reshape(-1,1), y_train)
    y_test_corr = scaler.predict(y_test_pred.reshape(-1,1))

    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_test_corr)),
        "MAE": mean_absolute_error(y_test, y_test_corr),
        "NSE": nse(y_test, y_test_corr),
        "KGE": kge(y_test, y_test_corr)
    }

    print("\nTEST METRICS")
    for k,v in metrics.items():
        print(f"{k}: {v:.3f}")

    # ================= Hydrograph plot =================
    test_dates = df.index[test_idx]

    plt.figure(figsize=(10,5))
    plt.plot(test_dates, y_test, label="Observed", marker='o')
    plt.plot(test_dates, y_test_pred, label="LSTM Raw", marker='x')
    plt.plot(test_dates, y_test_corr, label="LSTM + Linear-Scaling", marker='s')
    plt.title("Observed vs Predicted Monthly Flow – River Bosna")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gs_lstm_forecast_test_period.png", dpi=300)
    plt.close()

    # ================= Permutation Feature Importance =================
    feature_names = df.drop(columns=[target_col]).columns

    fi = permutation_importance_lstm(
        model=model,
        X=X_test_lstm,
        y=y_test,
        feature_names=feature_names,
        metric_fn=nse,
        n_repeats=5
    )

    # Compute feature direction: correlation between feature and prediction
    directions = []
    for i, col in enumerate(feature_names):
        corr = np.corrcoef(X_test[:, i], y_test_corr.flatten())[0,1]
        directions.append(corr)

    fi["direction"] = directions
    fi["color"] = fi["direction"].apply(lambda x: "green" if x >= 0 else "red")

    # Keep only top 5 most important features
    fi_top5 = fi.sort_values("importance", ascending=False).head(5)
    fi_top5 = fi_top5.sort_values("importance", ascending=True)  # for nicer horizontal plot

    # Plot
    plt.figure(figsize=(8,6))
    sns.barplot(
        x="importance", 
        y="feature", 
        data=fi_top5,
        palette=fi_top5["color"]
    )
    plt.xlabel("NSE decrease after permutation")
    plt.ylabel("Feature")
    plt.title("Top 5 LSTM Feature Importance with Direction")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gs_feature_importance_lstm_top5.png", dpi=300)
    plt.close()
    return metrics

# =====================================================
# Load data
# =====================================================
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

dfs = [
    padavine_doboj, padavine_tuzla, padavine_zenica,
    padavine_sarajevo, padavine_bjelasnica,
    temp_doboj, temp_tuzla, temp_zenica,
    temp_sarajevo, temp_bjelasnica,
    proticaj
]

df_merged = reduce(lambda l, r: pd.merge(l, r, on="date", how="inner"), dfs)

# =====================================================
# Run
# =====================================================
metrics = run_lstm_with_nse_tuning(df_merged)

