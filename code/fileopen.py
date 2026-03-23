import h5py

file = h5py.File("D:\crime\Spatio-Temporal-Crime-Hotspot-Prediction\Data\PreprocessedDatasets\8_features.h5","r")
print(list(file.keys()))

data = file["features"]

print(data.shape)
print(data[0])