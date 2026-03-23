import pickle

with open("D:\crime\Spatio-Temporal-Crime-Hotspot-Prediction\Data\ModelWeights\TimeSeriesModel_burglary.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data)
print(data.summary())