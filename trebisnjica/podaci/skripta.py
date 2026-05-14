import pandas as pd
import numpy as np
import os


# ==================== Deskriptivna analiza =============================================

# perticipation
stations = ["Berkovici","Bileca","Gacko","Grancarevo",  "Meka_gruda", "Mosko", "Nevesinje", "Odzak_rast", "Stepen"]
precip_data = {}

for s in stations:
    df = pd.read_excel(f"padavine/{s}.xlsx")
    df = df.rename(columns={df.columns[0]:"Year"})
    precip_data[s] = df
    
    
    
#temperature:
stations_temp = [ "Bileca","Gacko" ]
temp_data = {}

for s in stations_temp:
    df = pd.read_excel(f"temperature/{s}.xlsx")
    df = df.rename(columns={df.columns[0]:"Year"})
    temp_data[s] = df
    
#print(precip_data)
#print(temp_data["Bileca"])

## year perticipation:

annual_precip = pd.DataFrame()
months = ["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII"]

for s in stations:

    df = precip_data[s]

    annual_precip[s] = df[months].sum(axis=1)

annual_precip["Year"] = precip_data[stations[0]]["Year"]
annual_precip = annual_precip.set_index("Year")

#year temp:
annual_temp = pd.DataFrame()
for s in stations_temp:
    
    df = temp_data[s]
    annual_temp[s] = df[months].mean(axis=1)

annual_temp["Year"] = temp_data[stations_temp[0]]["Year"]
annual_temp = annual_temp.set_index("Year")

#print(annual_temp)

#4. Trend padavina (Mann-Kendall test)

import pymannkendall as mk

print("========================= TREND precip ============================= \n")

trend_results_precip = {}
for s in stations:
    result = mk.original_test(annual_precip[s])
    trend_results_precip[s] = result
    print(s, result)
    

print("========================= TREND temperature ============================= \n")
trend_results_temp = {}

for s in stations_temp:
    result = mk.original_test(annual_temp[s])
    trend_results_temp[s] = result
    print(s, result)  
    
#mk.original_test(annual_temp)    
#print(s, result)  
    
## Indeks sus:

from scipy.stats import gamma, norm

def calculate_spi(series):

    shape, loc, scale = gamma.fit(series, floc=0)
    cdf = gamma.cdf(series, shape, loc=loc, scale=scale)
    spi = norm.ppf(cdf)

    return spi

spi_data = {}

for s in stations:
    spi_data[s] = calculate_spi(annual_precip[s])
    
    
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.plot(annual_precip.index, spi_data["Bileca"])
plt.axhline(0)
plt.axhline(-1, linestyle="--")
plt.axhline(-2, linestyle="--")

plt.title("SPI indeks suše - Bileća")
plt.show()


## Prostorna varijabilnost padavina

import seaborn as sns

corr = annual_precip.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Korelacija padavina između stanica")
plt.show()


### PCA analysis of percipitation:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(annual_precip)

pca = PCA()
components = pca.fit_transform(X)
print(pca.explained_variance_ratio_)

# graph
plt.scatter(components[:,0], components[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA analiza padavina")
plt.show()

# Trend percipitation:

for s in stations:
    #break
    plt.figure()
    plt.plot(annual_precip.index, annual_precip[s])
    z = np.polyfit(annual_precip.index, annual_precip[s],1)#TODO: ML model -- ekstremi za Bosna river basin 
    p = np.poly1d(z)

    plt.plot(annual_precip.index, p(annual_precip.index))

    plt.title(s)
    plt.show()


## Seasonal percipitation:

winter = ["XII","I","II"]
summer = ["VI","VII","VIII"]

seasonal = {}

for s in stations:

    df = precip_data[s]

    seasonal[s] = {
        "winter": df[winter].sum(axis=1),
        "summer": df[summer].sum(axis=1)
    }
#Trend opet raditi Mann-Kendall: TODO
