# Smoke Tests

## Quick Test for Neuron Scripts

This directory contains a simple smoke test to verify that `scripts/neurons/train_models.py` and `scripts/neurons/estimate_pws.py` work correctly.

### Usage

Run from the project root:

```bash
./tests/test_neurons_scripts.sh
```

### What It Tests

1. **Training** - Trains a minimal RNN model on neuron 0 (2 epochs, hidden_size=4)
2. **PWS Estimation** - Runs PWS estimation with fast parameters (N=20, M=64, seq_len=15)
3. **Output Validation** - Checks that outputs are created and have the expected structure

### Expected Runtime

~1-2 minutes total

### Requirements

- The neuron dataset must exist at: `/data/clusterusers/reinhardt/containerhome/ml-pws/data/barmovie0113extended.data`
- Python environment with all dependencies installed (PyTorch, torchsde, mpi4py, etc.)

### Test Parameters

**Training (vs. production):**
- Epochs: 2 (vs. 50)
- Hidden size: 4 (vs. varies)
- Num layers: 1 (vs. varies)
- Single neuron only

**PWS Estimation (vs. production):**
- N: 20 trajectories (vs. 400)
- M: 64 particles (vs. 2048)
- seq_len: 15 timesteps (vs. 100)

### Output

Test artifacts are written to `tests/smoke_test_output/`:
- `models/test_neuron_0.pth` - Trained model checkpoint
- `pws_result.json` - PWS estimation results
- Config files

This directory is cleaned and recreated on each test run.
