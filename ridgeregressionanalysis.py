import pandas as pd
from numpy import arange
from pandas import read_csv
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
import matplotlib as plt

# Uses pandas library to import raw copper export data, copper prices, and oil prices.
exportRaw = pd.read_csv('yearly-export-per-country.csv')
copperRaw = pd.read_csv('copper-prices-historical-chart-data.csv')
oilRaw = pd.read_csv('oil_prices.csv')

# Limiting scope to greatest exporters of copper & aggregates export data
exportRaw = exportRaw.query(" CountryName == 'Chile' | CountryName == 'Peru' | CountryName == 'Australia' | CountryName == 'Canada' | CountryName == 'Mexico'")
exportVolume = exportRaw.sum(axis=0)
exportVolume = exportVolume.drop(["CountryName","CountryCode","IndicatorName","IndicatorCode"],axis = 0)
exportVolume = exportVolume[exportVolume.values != 0]
exportDict0 = exportVolume.to_dict()
exportDict = dict()

for key in exportDict0.keys():
    exportDict[int(key)] = exportDict0.get(key)


# Standardizing time data for oil prices & calculates average cost of oil per year
oilRaw = oilRaw.dropna()
oilRaw['date'] = pd.to_datetime(oilRaw.date)
oilPrices = oilRaw.groupby(oilRaw.date.dt.year)['price'].mean()
oilDict = oilPrices.to_dict()

# Standardizing time data for copper prices & calculates average cost of copper per year
copperRaw = copperRaw.dropna()
copperRaw['date'] = pd.to_datetime(copperRaw.date)
copperPrices = copperRaw.groupby(copperRaw.date.dt.year)['price'].mean()
copperDict = copperPrices.to_dict()

# Combines the data in a singular Data Frame
yearList = []
for i in range(1959,2022):
    yearList.append(i)

combinedData = pd.DataFrame(columns=['Year','Oil_Prices','Export_Volume','Copper_Prices'])
combinedData = combinedData.assign(Year=yearList)
combinedData['Copper_Prices'] = combinedData["Year"].map(copperDict)
combinedData['Oil_Prices'] = combinedData["Year"].map(oilDict)
combinedData['Export_Volume'] = combinedData["Year"].map(exportDict)
combinedData = combinedData.dropna()
writer = pd.ExcelWriter('CombinedData.xlsx')
combinedData.to_excel(writer)
writer.save()
# print(combinedData.to_string())

# Ridge Regression
data = combinedData.values
x, y = data[:, :-1], data[:, -1]
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# Define model
model = RidgeCV(alphas=arange(0.01, 1, 10.0), cv=cv, scoring='neg_mean_absolute_error')
# # Fit model
model.fit(x, y)
# Summarize chosen configuration
# print('alpha: %f' % model.alpha_)
coefVector = model.coef_



# print(x)
# print(y)

# print(combinedData.to_string())

