import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from functools import reduce
from convert import convert_data

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def kge(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    r = np.corrcoef(y_true, y_pred)[0,1]                  # correlation
    beta = np.mean(y_pred) / np.mean(y_true)             # bias ratio
    gamma = (np.std(y_pred)/np.mean(y_pred)) / (np.std(y_true)/np.mean(y_true))  # variability ratio
    return 1 - np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)


# ================================
# 1) Lag feature function
# ================================
def add_lag_features(df, lags_P=[1,2,3], lags_T=[1,2], lag_Q=1):
    df = df.copy()
    P_cols = [c for c in df.columns if c.startswith("P_")]
    T_cols = [c for c in df.columns if c.startswith("T_")]

    # Precipitation lags
    for lag in lags_P:
        for col in P_cols:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Temperature lags
    for lag in lags_T:
        for col in T_cols:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Discharge lag (autoregressive)
    if "Q_proticaj" in df.columns:
        df["Q_lag1"] = df["Q_proticaj"].shift(lag_Q)

    return df


# ================================
# 2) LSTM forecast function
# ================================
def run_lstm_forecast_with_lags(df, target_col="Q_proticaj",
                                max_train_year=2010, test_start_year=2011,
                                lstm_units=100, epochs=1000, batch_size=32):

    # 0) Ensure datetime index
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    df = df.sort_index()

    # 1) Add lag features
    df = add_lag_features(df,
                          lags_P=[1,2,3],
                          lags_T=[1,2],
                          lag_Q=1)

    # Drop rows with missing lags
    df = df.dropna()

    # 2) Feature / target
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # 3) Train / test split
    train_idx = df.index.year <= max_train_year
    test_idx  = df.index.year >= test_start_year

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # 4) Reshape for LSTM: [samples, timesteps=1, features]
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # 5) Build LSTM model
    model = Sequential([
        LSTM(lstm_units, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # 6) Train
    model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # 7) Predict
    y_pred = model.predict(X_test_lstm).flatten()
    y_train_pred = model.predict(X_train_lstm).flatten()

    # 8) Linear-scaling bias correction
    scaler = LinearRegression()
    scaler.fit(y_train_pred.reshape(-1,1), y_train)
    a, b = scaler.coef_[0], scaler.intercept_
    print(f"Linear-scaling: a={a:.3f}, b={b:.3f}")

    y_pred_corr = a * y_pred + b

    # 9) Metrics
    def nse(y_true, y_hat):
        return 1 - np.sum((y_true - y_hat)**2) / np.sum((y_true - y_true.mean())**2)

    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_corr)),
        "MAE":  mean_absolute_error(y_test, y_pred_corr),
        "NSE":  nse(y_test, y_pred_corr),
        "KGE":  kge(y_test, y_pred_corr)

    }

    print("\nFORECAST METRICS (1-step ahead)")
    for k,v in metrics.items():
        print(f"{k}: {v:.3f}")

    # 10) Plot
    plt.figure(figsize=(10,5))
    plt.plot(df.index[test_idx], y_test, label="Observed")
    plt.plot(df.index[test_idx], y_pred_corr, label="Forecast (bias-corrected)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return y_pred_corr, metrics


# ================================
# 3) Load and merge data
# ================================

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

# -- proticaj --
proticaj = convert_data("proticaj-proticaj.ods", prefix="Q")

# NAN VALUES TREATING  temp_zenica Climatological Monthly Mean Filling (BEST for 6-month gap)
temp_zenica['date'] = pd.to_datetime(temp_zenica['date'])
temp_zenica = temp_zenica.set_index('date')

temp_zenica['month'] = temp_zenica.index.month
monthly_mean = temp_zenica.groupby('month')['T_zenica'].mean()

mask = (temp_zenica.index >= '2017-07-01') & (temp_zenica.index <= '2017-12-01')
temp_zenica.loc[mask, 'T_zenica'] = temp_zenica.loc[mask].index.month.map(monthly_mean)


# Merge data
dfs = [
    padavine_doboj, padavine_tuzla, padavine_zenica,
    padavine_sarajevo, padavine_bjelasnica,
    temp_doboj, temp_tuzla, temp_zenica,
    temp_sarajevo, temp_bjelasnica,
    proticaj
]

df_merged = reduce(lambda left, right: pd.merge(left, right, on="date", how="inner"), dfs)
print(df_merged.head())

# ================================
# 4) Run LSTM forecast
# ================================

y_pred_corr, metrics = run_lstm_forecast_with_lags(df_merged)
