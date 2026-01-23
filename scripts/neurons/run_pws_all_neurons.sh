#!/bin/bash
# Run PWS estimation on all pre-trained neuron models
#
# This script iterates through all neuron models (neuron_0.pth to neuron_49.pth)
# and runs PWS estimation using estimate_pws.py, saving results to experiments/neurons_pws/

set -e  # Exit on error

# Configuration
MODEL_DIR="models"
OUTPUT_DIR="experiments/neurons_pws"
NUM_NEURONS=50

# PWS estimation parameters
N=400           # Number of trajectories to sample
M=2048          # Number of particles for marginal estimation
SEQ_LEN=100     # Sequence length in time bins

# Model architecture parameters (should match training configuration)
N_NEURONS=1
HIDDEN_SIZE=4
NUM_LAYERS=2
MODEL_TYPE="CNN"
KERNEL_SIZE=20

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Script directory (where estimate_pws.py is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting PWS estimation for $NUM_NEURONS neurons"
echo "Output directory: $OUTPUT_DIR"
echo "Parameters: N=$N, M=$M, seq_len=$SEQ_LEN"
echo ""

# Track progress
success_count=0
fail_count=0
failed_neurons=()

# Process each neuron
for neuron_id in $(seq 0 $((NUM_NEURONS - 1))); do
    model_file="$MODEL_DIR/neuron_${neuron_id}.pth"
    output_file="$OUTPUT_DIR/neuron_${neuron_id}_results.json"

    # Check if model exists
    if [ ! -f "$model_file" ]; then
        echo "Warning: Model not found at $model_file, skipping"
        ((fail_count++))
        failed_neurons+=($neuron_id)
        continue
    fi

    # Check if output already exists (skip if present)
    if [ -f "$output_file" ]; then
        echo "Neuron $neuron_id: Results already exist at $output_file, skipping"
        ((success_count++))
        continue
    fi

    echo "Processing neuron $neuron_id..."

    # Run PWS estimation
    if python "$SCRIPT_DIR/estimate_pws.py" \
        "$model_file" \
        --output "$output_file" \
        --n-neurons $N_NEURONS \
        --hidden-size $HIDDEN_SIZE \
        --num-layers $NUM_LAYERS \
        --model-type $MODEL_TYPE \
        --kernel-size $KERNEL_SIZE \
        --N $N \
        --M $M \
        --seq-len $SEQ_LEN; then
        ((success_count++))
        echo "✓ Neuron $neuron_id completed successfully"
    else
        ((fail_count++))
        failed_neurons+=($neuron_id)
        echo "✗ Neuron $neuron_id failed"
    fi

    echo ""
done

# Summary
echo "=========================================="
echo "PWS Estimation Summary"
echo "=========================================="
echo "Total neurons: $NUM_NEURONS"
echo "Successfully processed: $success_count"
echo "Failed: $fail_count"

if [ ${#failed_neurons[@]} -gt 0 ]; then
    echo "Failed neuron IDs: ${failed_neurons[*]}"
fi

echo ""
echo "Results saved to: $OUTPUT_DIR"
