import json
from pathlib import Path

PREFIX = Path(".")

runs = []

# runs.append({
#     "neurons": list(range(50)),
#     "name": "neurons_range(50)"
# })

for n in range(230):
    r = {
        "neurons": [n],
        "name": f"neuron_{n}",
        "output_dir": str(PREFIX / "experiments" / "neurons" / f"neuron_{n}"),
        "hidden_size": 40,
        "num_layers": 4,
        "kernel_size": 20,
    }
    runs.append(r)

config = {
    "dataset_path": str(PREFIX / "data" / "barmovie0113extended.data"),
    "model_path": str(PREFIX / "models"),
    "N": 1000,
    "M": 256,
    "seq_len": 200,
    "models": runs,
}

with open(PREFIX / "scripts" / "neurons" / "config.json", "w") as f:
    json.dump(config, f, indent=4)
