#!/usr/bin/env bash
# ==============================================================================
# Execution script for reproduction experiments on RTX 4090 Linux Server
# Usage:
#   bash run_experiments_4090.sh smoke
#   bash run_experiments_4090.sh end-to-end [workers] (default workers: 32)
#   bash run_experiments_4090.sh blackbox [workers]   (default workers: 8)
#   bash run_experiments_4090.sh leaky [workers]      (default workers: 8)
#   bash run_experiments_4090.sh whitebox [workers]   (default workers: 10)
# ==============================================================================

set -e

ENV_NAME="hardlabel"
CONDA_BASE=$(conda info --base 2>/dev/null || true)
if [ -n "$CONDA_BASE" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}" 2>/dev/null || true
fi

MODE="${1:-smoke}"
WORKERS="${2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

case "${MODE}" in
    smoke)
        echo "=== [Smoke Test] Verifying Oracle & Model Loading ==="
        python -c "
import numpy as np, scipy, torch, h5py
import os, sys
sys.path.insert(0, 'full_end_to_end')
import oracle
print('Oracle query label test (zeros):', oracle.label(np.zeros(32)))
print('Queries so far:', oracle.query_count())

import tensorflow as tf, keras
from pathlib import Path
for p in Path('data').glob('*.keras'):
    m = tf.keras.models.load_model(p, compile=False)
    print(f'Successfully loaded model {p.name}: input shape {m.input_shape}')
"
        echo "=== Smoke Test PASSED ==="
        ;;

    end-to-end)
        NUM_WORKERS="${WORKERS:-32}"
        echo "=== [Experiment 1] Full End-to-End Extraction ==="
        echo "Running with ${NUM_WORKERS} workers on CPU..."
        cd full_end_to_end
        # Cleans partial results if requested, or keeps resuming
        python run.py "${NUM_WORKERS}"
        echo "=== End-to-End Extraction and Validation Completed! ==="
        ;;

    blackbox)
        NUM_WORKERS="${WORKERS:-8}"
        echo "=== [Experiment 2] Blackbox Sign Recovery ==="
        cd sign_recovery
        python sign_recovery_blackbox.py --model unitary_32_32x3_10_float64 --layer 1-3 --neuron 0-7 --j "${NUM_WORKERS}"
        echo "Parsing results into tables..."
        python create_tables.py
        ;;

    leaky)
        NUM_WORKERS="${WORKERS:-8}"
        echo "=== [Experiment 3] Leaky ReLU Sign Recovery ==="
        cd sign_recovery
        python sign_recovery_leaky.py --model unitary_leaky_32_32x3_10_float64 --j "${NUM_WORKERS}"
        ;;

    whitebox)
        NUM_WORKERS="${WORKERS:-10}"
        echo "=== [Experiment 4] CIFAR-10 Whitebox Sign Recovery ==="
        cd sign_recovery
        if [ ! -d "../data/dual_points_cifar10_3x256_64_10_float64" ]; then
            echo "[WARNING] Precomputed dual points directory not found in data/!"
            echo "Please download precomputed dual points or run generators/dual_generator_whitebox.py first."
        fi
        python sign_recovery_whitebox.py --model cifar10_3x256_64_10_float64 --layer 1 --neuron 0-15 --j "${NUM_WORKERS}"
        python create_tables.py
        ;;

    *)
        echo "Unknown mode: ${MODE}"
        echo "Available options: smoke | end-to-end | blackbox | leaky | whitebox"
        exit 1
        ;;
esac

