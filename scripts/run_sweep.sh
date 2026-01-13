#!/bin/bash
# Run a wandb sweep with multiple parallel agents
# Usage: ./scripts/run_sweep.sh [num_agents] [sweep_config]

NUM_AGENTS=${1:-2}
SWEEP_CONFIG=${2:-sweep.yaml}

# Create the sweep and capture the full sweep path
echo "Creating sweep from $SWEEP_CONFIG..."
SWEEP_OUTPUT=$(wandb sweep "$SWEEP_CONFIG" 2>&1)
echo "$SWEEP_OUTPUT"

# Extract full sweep path (entity/project/sweep_id) from "wandb agent entity/project/id"
SWEEP_PATH=$(echo "$SWEEP_OUTPUT" | grep -oE 'wandb agent [^ ]+' | awk '{print $3}')

if [ -z "$SWEEP_PATH" ]; then
  echo "Failed to extract sweep path from output"
  exit 1
fi

echo ""
echo "Sweep path: $SWEEP_PATH"
echo "Starting $NUM_AGENTS parallel agents..."

# Start agents in background
PIDS=()
for i in $(seq 1 $NUM_AGENTS); do
  echo "  Starting agent $i..."
  wandb agent "$SWEEP_PATH" &
  PIDS+=($!)
  sleep 1 # Stagger agent starts slightly
done

echo ""
echo "All agents started. PIDs: ${PIDS[*]}"
# Convert entity/project/id to entity/project/sweeps/id
SWEEP_URL=$(echo "$SWEEP_PATH" | sed 's|\([^/]*/[^/]*\)/|\1/sweeps/|')
echo "Sweep URL: https://wandb.ai/$SWEEP_URL"
echo ""
echo "Press Ctrl+C to stop all agents"

# Wait for all agents and handle Ctrl+C
trap 'echo "Stopping agents..."; kill ${PIDS[*]} 2>/dev/null; exit 0' INT TERM
wait
