import json
import os.path

PREFIX = os.path.abspath(".")

runs = []

runs.append({
    "neurons": list(range(50)),
    "name": "neurons_range(50)"
})

for n in range(50):
    r = {
            "neurons": [n],
            "name": f"neuron_{n}"
    }
    runs.append(r)

config = {
        "dataset_path": PREFIX + "/data/barmovie0113extended.data",
        "model_path": PREFIX + "/models/",
        "models": runs
        }

with open(PREFIX + "/scripts/neurons/config.json", "w") as f:
    json.dump(config, f)
