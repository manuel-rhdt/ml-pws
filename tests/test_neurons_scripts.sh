#!/bin/bash
# Smoke test for neuron scripts (train_models.py and estimate_pws.py)
# Tests that both scripts run without errors using minimal parameters

set -e  # Exit on any error

echo "=== Neuron Scripts Smoke Test ==="
echo

# Setup
PROJECT_ROOT="$(pwd)"
TEST_DIR="$PROJECT_ROOT/tests/smoke_test_output"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR/models"

# Check if dataset exists
DATASET="data/barmovie0113extended.data"
if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found at $DATASET"
    echo "Skipping smoke test (requires real neuron data)"
    exit 1
fi

# Get absolute path to dataset
DATASET_ABS="$PROJECT_ROOT/$DATASET"

# 1. Create minimal training config
echo "Creating training config..."
cat > "$TEST_DIR/train_config.json" <<EOF
{
  "dataset_path": "$DATASET_ABS",
  "model_path": "$TEST_DIR/models/",
  "models": [
    {"neurons": [0], "name": "test_neuron_0"}
  ]
}
EOF

# 2. Run training (minimal epochs)
echo "Running train_models.py (50 epochs - hardcoded, 1 neuron)..."
cd scripts/neurons
python train_models.py "$TEST_DIR/train_config.json" \
    || { echo "FAILED: train_models.py crashed"; exit 1; }

cd "$PROJECT_ROOT"

# Check training output
if [ ! -f "$TEST_DIR/models/test_neuron_0.pth" ]; then
    echo "FAILED: Model checkpoint not created"
    exit 1
fi
echo "✓ Training completed successfully"
echo

# 3. Create minimal PWS config
echo "Creating PWS config..."
cat > "$TEST_DIR/pws_config.json" <<EOF
{
  "n_neurons": 1,
  "hidden_size": 4,
  "num_layers": 2,
  "model_type": "CNN",
  "kernel_size": 20,
  "N": 20,
  "M": 64,
  "seq_len": 15,
  "models": [
    {
      "neuron_id": 0,
      "model_path": "$TEST_DIR/models/test_neuron_0.pth",
      "output_path": "$TEST_DIR/pws_result.json"
    }
  ]
}
EOF

# 4. Run PWS estimation
echo "Running estimate_pws.py (N=20, M=64, seq_len=15)..."
cd scripts/neurons
python estimate_pws.py "$TEST_DIR/pws_config.json" \
    || { echo "FAILED: estimate_pws.py crashed"; exit 1; }
cd "$PROJECT_ROOT"

# 5. Validate outputs
echo
echo "Validating outputs..."

# Check PWS output exists
if [ ! -f "$TEST_DIR/pws_result.json" ]; then
    echo "FAILED: PWS result not created"
    exit 1
fi

# Validate JSON is parseable
python -c "import json; json.load(open('$TEST_DIR/pws_result.json'))" \
    || { echo "FAILED: Invalid JSON output"; exit 1; }

# Check for required fields
python -c "
import json
with open('$TEST_DIR/pws_result.json') as f:
    result = json.load(f)
    assert 'pws_result' in result, 'Missing pws_result field'
    assert 'mutual_information' in result['pws_result'], 'Missing mutual_information field'
    assert len(result['pws_result']['mutual_information']) == 15, f'Expected 15 timesteps, got {len(result[\"pws_result\"][\"mutual_information\"])}'
    print('✓ PWS result has expected structure')
" || { echo "FAILED: PWS result missing required fields"; exit 1; }

echo
echo "=== ALL TESTS PASSED ==="
echo "Training output: $TEST_DIR/models/test_neuron_0.pth"
echo "PWS result: $TEST_DIR/pws_result.json"
echo
echo "Final MI estimate:"
python -c "
import json
with open('$TEST_DIR/pws_result.json') as f:
    result = json.load(f)
    mi = result['pws_result']['mutual_information'][-1]
    mi_std = result['pws_result']['mutual_information_std'][-1]
    print(f'  MI = {mi:.4f} ± {mi_std:.4f}')
"
