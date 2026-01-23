import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression


def kge(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    r = np.corrcoef(y_true, y_pred)[0,1]                  # correlation
    beta = np.mean(y_pred) / np.mean(y_true)             # bias ratio
    gamma = (np.std(y_pred)/np.mean(y_pred)) / (np.std(y_true)/np.mean(y_true))  # variability ratio
    return 1 - np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)

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
    
def run_xgboost_forecast_with_lags(df, target_col="Q_proticaj",
                                   max_train_year=2010, test_start_year=2011):

    # -------------------------------------------------
    # 0) Ensure datetime index
    # -------------------------------------------------
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    df = df.sort_index()

    # -------------------------------------------------
    # 1) Add lag features
    # -------------------------------------------------
    df = add_lag_features(df,
                          lags_P=[1,2,3],
                          lags_T=[1,2],
                          lag_Q=1)

    # Drop rows with missing lags
    df = df.dropna()

    # -------------------------------------------------
    # 2) Feature / target
    # -------------------------------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # -------------------------------------------------
    # 3) Train / test split
    # -------------------------------------------------
    train_idx = df.index.year <= max_train_year
    test_idx  = df.index.year >= test_start_year

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test,  y_test  = X.loc[test_idx],  y.loc[test_idx]

    # -------------------------------------------------
    # 4) Model
    # -------------------------------------------------
    model = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    # -------------------------------------------------
    # 5) Prediction (1-step-ahead)
    # -------------------------------------------------
    y_pred = model.predict(X_test)

    # -------------------------------------------------
    # 6) Linear-scaling bias correction applied
    # -------------------------------------------------
    scaler = LinearRegression()
    scaler.fit(model.predict(X_train).reshape(-1,1), y_train.values)

    a, b = scaler.coef_[0], scaler.intercept_
    print(f"Linear-scaling: a={a:.3f}, b={b:.3f}")

    y_pred_corr = a * y_pred + b

    # -------------------------------------------------
    # 7) Metrics
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 8) Plot
    # -------------------------------------------------
    plt.figure(figsize=(10,5))
    plt.plot(y_test.index, y_test, label="Observed")
    plt.plot(y_test.index, y_pred_corr, label="Forecast (bias-corrected)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return y_pred_corr, metrics



def run_xgboost_with_linear_scaling(df, target_col="Q_proticaj", max_train_year=2010, test_start_year=2011):
    """
    Run XGBoost on monthly hydrological data with linear-scaling bias correction.

    Args:
        df (pd.DataFrame): DataFrame with datetime index and features including target_col.
        target_col (str): Column name of target (e.g., river flow).
        max_train_year (int): Maximum year for training period.
        test_start_year (int): Start year of test period.

    Returns:
        y_pred_corrected (np.array): Bias-corrected predictions.
        metrics (dict): RMSE, MAE, NSE.
    """
    
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex or a 'date' column")

    df = df.sort_index()

    # 1) Feature/target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # -------------------------------------------------
    # 2) Train / Test split (NOW SAFE)
    # -------------------------------------------------
    train_idx = df.index.year <= max_train_year
    test_idx  = df.index.year >= test_start_year
    

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test   = X.loc[test_idx],  y.loc[test_idx]

    # 3) XGBoost model
    model = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X_train, y_train)

    # 4) Raw predictions
    y_pred = model.predict(X_test)

    # 5) Linear-scaling bias correction
    scaler = LinearRegression()
    scaler.fit(model.predict(X_train).reshape(-1,1), y_train.values)
    a, b = scaler.coef_[0], scaler.intercept_
    print(f"Linear-scaling coefficients: a={a:.3f}, b={b:.3f}")

    y_pred_corrected = a * y_pred + b

    # 6) Metrics
    def nse(y_true, y_pred):
        return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)

    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_corrected)),
        "MAE" : mean_absolute_error(y_test, y_pred_corrected),
        "NSE" : nse(y_test.values, y_pred_corrected)
    }

    print("\nFINAL TEST METRICS (After Linear-Scaling):")
    for k,v in metrics.items():
        print(f"{k}: {v:.3f}")

    # 7) Plots
    plt.figure(figsize=(10,5))
    plt.plot(y_test.index, y_test.values, label="Observed", marker='o')
    plt.plot(y_test.index, y_pred, label="XGBoost Raw", marker='x')
    plt.plot(y_test.index, y_pred_corrected, label="XGBoost + Linear-Scaling", marker='s')
    plt.title("Observed vs Predicted Monthly Flow – River Bosna")
    plt.xlabel("Date")
    plt.ylabel("Flow (m³/s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Feature importance plot (top 5)
    imp = pd.Series(model.feature_importances_, index=X.columns)
    top_imp = imp.sort_values(ascending=True).tail(5)
    plt.figure(figsize=(8,5))
    top_imp.plot(kind="barh", color="skyblue")
    plt.title("Top 5 Feature Importances – XGBoost")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return y_pred_corrected, metrics



from convert import convert_data
import time 
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from functools import reduce


# 0) PRIPREMA PODATAKA:

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

y_pred_corr, metrics = run_xgboost_forecast_with_lags(df_merged)    #run_xgboost_with_linear_scaling(df_merged)
