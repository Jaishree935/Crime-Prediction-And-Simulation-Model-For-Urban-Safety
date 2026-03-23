import torch


model = torch.load(
    r"D:\crime\Spatio-Temporal-Crime-Hotspot-Prediction\Data\ModelWeights\BestModel__bs-(16)_threshold-(0.5)_weights-([1, 30]).pt",
    map_location=torch.device('cpu')
)

print(model)

print(model.keys())

print(type(model))
