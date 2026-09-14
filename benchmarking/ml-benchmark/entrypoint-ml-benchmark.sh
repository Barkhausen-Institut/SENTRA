#!/bin/bash
# ML benchmark image: bash|sh, *.py, YAML driver (--config), or SENTRA_OCCLUM_RUN=1.
set -euo pipefail
export PYTHONPATH=/workspace/node:/workspace/client
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
cd /workspace/node

if [[ "${1:-}" == "bash" || "${1:-}" == "sh" ]]; then
  exec "$@"
fi

if [[ "${1:-}" == "sgx" ]]; then
  cd /occlum-instance
  export SENTRA_PREWARM_TRAIN_UNPACK="${SENTRA_PREWARM_TRAIN_UNPACK:-1}"
  export SENTRA_BARRIER_RESEND_S="${SENTRA_BARRIER_RESEND_S:-2}"
  export SENTRA_STRICT_UNPACK_BARRIER="${SENTRA_STRICT_UNPACK_BARRIER:-1}"
  exec occlum run /bin/enclave_run_script.sh $2 $3
fi

if [[ "${1:-}" == *.py ]]; then
  exec /python-occlum/bin/python "$@"
fi

exec /python-occlum/bin/python /usr/local/bin/run_benchmark_from_config.py "$@"
