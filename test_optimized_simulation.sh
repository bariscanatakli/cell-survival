#!/bin/bash

echo "===== Testing Cell Survival Simulation with Optimized Batch Processing ====="
echo "Starting test run with 5 episodes, 10000 steps, and 200 cells..."
echo ""

# Run with optimized batch processing
SECONDS=0
python3 src/main.py --episodes 5 --max-steps 10000 --world-size 2048 --num-cells 200 \
  --num-foods 800 --num-hazards 30 --batch-size 128 --no-render

ELAPSED="$SECONDS"
echo ""
echo "====== Performance Results ======"
echo "Simulation completed in $(($ELAPSED/60)) minutes and $(($ELAPSED%60)) seconds"
echo "This is the new optimized batch processing version."
echo ""

chmod +x test_optimized_simulation.sh
