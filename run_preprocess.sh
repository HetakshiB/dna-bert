#!/usr/bin/env bash
# run_preprocess.sh - Execute the Multimodal CRISPR Preprocessing Pipeline
set -e

# Determine script and project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Allow custom data directory via CRISPR_DATA_DIR env variable
export CRISPR_DATA_DIR="${CRISPR_DATA_DIR:-$SCRIPT_DIR/data}"

echo "==========================================================================="
echo "Running Multimodal CRISPR Preprocessing Pipeline"
echo "Project Root: $SCRIPT_DIR"
echo "Data Root   : $CRISPR_DATA_DIR"
echo "==========================================================================="

# Activate virtual environment if present
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [ -f "$SCRIPT_DIR/.venv/Scripts/activate" ]; then
    source "$SCRIPT_DIR/.venv/Scripts/activate"
fi

python utils/preprocess.py "$@"
