#!/usr/bin/env bash
# ==============================================================================
# Setup script for hard-label-dnn-extraction on Linux server with RTX 4090
# Python: 3.11 | PyTorch: CUDA 12.1+ | TensorFlow / Keras 3
# ==============================================================================

set -e

ENV_NAME="hardlabel"
PYTHON_VERSION="3.11"

echo "=== [1/5] Checking Conda Installation ==="
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda not found in PATH. Please install Miniconda/Anaconda or initialize conda."
    exit 1
fi

echo "Conda found: $(conda --version)"

echo "=== [2/5] Creating Conda Environment (${ENV_NAME}, Python ${PYTHON_VERSION}) ==="
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Environment ${ENV_NAME} already exists. Skipping creation."
else
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# Activate conda environment in bash script
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "=== [3/5] Upgrading pip ==="
python -m pip install --upgrade pip

# CUDA wheel selection: default cu118 for maximum compatibility with CUDA 12.0 driver, or cu121 if preferred
CUDA_WHEEL="${1:-cu118}"

echo "=== [4/5] Installing PyTorch with CUDA (${CUDA_WHEEL}) Support for RTX 4090 ==="
echo "Note: If your nvidia-smi shows CUDA 12.0, 'cu118' is 100% compatible (driver backwards compatible) and supports RTX 4090 (sm_89)."
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"

echo "=== [5/5] Installing Scientific Computing & Keras 3 Dependencies ==="
# Lock numpy to 1.26.4 and ml-dtypes to <=0.5.1 to prevent pip from fetching numpy 2.x which breaks scipy 1.13.1
pip install "numpy==1.26.4" "scipy==1.13.1" "h5py==3.11.0" "ml-dtypes<=0.5.1"
pip install "tensorflow==2.20.0" "keras==3.15.1" "pandas>=2.2.0" "matplotlib>=3.9.0" "tabulate>=0.9.0" "numpy==1.26.4" "ml-dtypes<=0.5.1"

echo ""
echo "=== Environment Verification ==="
python -c "
import torch
print('PyTorch Version      :', torch.__version__)
print('CUDA Available       :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device Name      :', torch.cuda.get_device_name(0))
    print('Device Capability    :', torch.cuda.get_device_capability(0))
"

python -c "
import tensorflow as tf, keras
print('TensorFlow Version   :', tf.__version__)
print('Keras Version        :', keras.__version__)
"

echo ""
echo ">>> Conda environment '${ENV_NAME}' successfully prepared!"
echo ">>> Run 'conda activate ${ENV_NAME}' before starting experiments."

