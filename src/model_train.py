# =========================================================
# XGBOOST PREDIKCIJA MJESECNOG PROTICAJA – RIJEKA BOSNA
# Rolling / Expanding Time Validation
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from convert import convert_data

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


# ---------------------------------------------------------
# 1) UČITAVANJE PODATAKA
# ---------------------------------------------------------
# CSV kolone:
# date,T1,T2,T3,T4,T5,P1,P2,P3,P4,P5,Q

df = df_merged#pd.read_csv("bosna_monthly.csv", parse_dates=["date"])
df = df.set_index("date").sort_index()

# ---------------------------------------------------------
# 2) FEATURE ENGINEERING
# ---------------------------------------------------------

# --- Sezonalnost (ciklični mjesec)
df["month"] = df.index.month
df["sin_m"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_m"] = np.cos(2 * np.pi * df["month"] / 12)

# --- Lagovi temperature i padavina
LAGS = [1, 2, 3]

for lag in LAGS:
    for i in [ "doboj", "tuzla", "zenica", "sarajevo", "bjelasnica" ]:
        df[f"P_{i}_lag{lag}"] = df[f"P_{i}"].shift(lag)
        df[f"T_{i}_lag{lag}"] = df[f"T_{i}"].shift(lag)

# --- Lag proticaja (dinamički efekat)
df["Q_lag1"] = df["Q_proticaj"].shift(1)

# --- Agregacije po slivu
df["P_mean"] = df[[f"P_{i}" for i in [ "doboj", "tuzla", "zenica", "sarajevo", "bjelasnica" ] ]].mean(axis=1)
df["P_max"]  = df[[f"P_{i}" for i in  [ "doboj", "tuzla", "zenica", "sarajevo", "bjelasnica" ] ] ].max(axis=1)

# --- Ukloni NA (zbog lagova)
df = df.dropna()

# ---------------------------------------------------------
# 3) FEATURE / TARGET
# ---------------------------------------------------------
X = df.drop(columns=["Q_proticaj"])
y = df["Q_proticaj"]

# ---------------------------------------------------------
# 4) METRIKE
# ---------------------------------------------------------
def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)

# ---------------------------------------------------------
# 5) XGBOOST MODEL
# ---------------------------------------------------------
def get_model():
    return XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

# ---------------------------------------------------------
# 6) ROLLING / EXPANDING TIME VALIDATION
# ---------------------------------------------------------
results = []

for split_year in range(1980, 2010):
    train_idx = df.index.year <= split_year
    test_idx  = df.index.year == split_year + 1

X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_test, y_test   = X.loc[test_idx],  y.loc[test_idx]

model = get_model()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results.append({
"Test_Year": split_year + 1,
"RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
"MAE": mean_absolute_error(y_test, y_pred),
"NSE": nse(y_test.values, y_pred)
})

cv_results = pd.DataFrame(results)

print("\nROLLING VALIDATION SUMMARY")
print(cv_results.describe())

# ---------------------------------------------------------
# 7) FINALNI MODEL (1961–2010 → test 2011–2020)
# ---------------------------------------------------------
train_idx = df.index.year <= 2010
test_idx  = df.index.year >= 2011

X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_test, y_test   = X.loc[test_idx],  y.loc[test_idx]

final_model = get_model()
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

print("\nFINAL TEST (2011–2020)")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE :", mean_absolute_error(y_test, y_pred))
print("NSE :", nse(y_test.values, y_pred))

# ---------------------------------------------------------
# 8) FEATURE IMPORTANCE (za sanity check)
# ---------------------------------------------------------
imp = pd.Series(final_model.feature_importances_, index=X.columns)
imp = imp.sort_values(ascending=False).head(5)

plt.figure(figsize=(8,5))
imp.plot(kind="barh")
plt.title("Top 15 Feature Importances – XGBoost")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("fig_river_bosna_shap.png")
plt.show()  

## --display 
# ---- Observed vs Predicted ----
plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test.values, label="Observed", marker='o')
plt.plot(y_test.index, y_pred, label="Predicted", marker='x')
plt.title("Observed vs Predicted Monthly Flow – River Bosna")
plt.xlabel("Date")
plt.ylabel("Flow (m³/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig_river_bosna_prediction.png")
plt.show()

