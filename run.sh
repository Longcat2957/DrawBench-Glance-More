#!/bin/bash

# All models execution script
# Usage: ./run_all_models.sh [additional arguments]

# Stop script on error
set -e

# Record script start time
echo "===== Script execution started: $(date) ====="

# Default arguments
DEFAULT_ARGS="--all-categories --num 100"

# Use additional arguments if provided, otherwise use default
if [ $# -eq 0 ]; then
    ARGS="$DEFAULT_ARGS"
else
    ARGS="$@"
fi

# Execute all models
echo "Running Sana_Sprint_1.6B_1024px_diffusers model..."
python main.py --repo-id "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers" $ARGS
echo "Sana_Sprint_1.6B_1024px_diffusers model execution completed"
echo ""

echo "Running Sana_Sprint_0.6B_1024px_diffusers model..."
python main.py --repo-id "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers" $ARGS
echo "Sana_Sprint_0.6B_1024px_diffusers model execution completed"
echo ""

echo "Running SANA1.5_4.8B_1024px_diffusers model..."
python main.py --repo-id "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers" $ARGS
echo "SANA1.5_4.8B_1024px_diffusers model execution completed"
echo ""

echo "Running SANA1.5_1.6B_1024px_diffusers model..."
python main.py --repo-id "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers" $ARGS
echo "SANA1.5_1.6B_1024px_diffusers model execution completed"
echo ""

# Record script end time
echo "===== Script execution completed: $(date) ====="
echo "All model executions completed."