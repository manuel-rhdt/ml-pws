from importlib import import_module
from itertools import product

from sacred.observers import MongoObserver

ex = import_module('01-model-size').ex

ex.observers.append(MongoObserver())

models = ['RNN', 'GRU', 'LSTM']
hidden_sizes = [1, 2, 4, 8, 16, 32]
layers = [1, 2]

for model, hidden_size, layer in product(models, hidden_sizes, layers):
    ex.run(config_updates={
        'model': model,
        'hidden_size': hidden_size,
        'layers': layer,
        'epochs': 1_000,
    })