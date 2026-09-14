#!/bin/bash
#/opt/occlum/start_aesm.sh
NO_SGX=false
args=()

for arg in "$@"; do
    case "$arg" in
        -no_sgx)
            NO_SGX=true
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done

set -- "${args[@]}"

wait-for-it -t 60 sentra-backend:8888

if ! $NO_SGX; then
    occlum run /bin/enclave_run_script.sh "$@"
else
    /bin/enclave_run_script.sh "$@"
fi
