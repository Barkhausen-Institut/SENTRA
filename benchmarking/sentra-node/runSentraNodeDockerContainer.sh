#!/bin/sh
docker run --rm -it --device /dev/sgx_enclave:/dev/sgx_enclave --device /dev/sgx_provision:/dev/sgx_provision registry.tdp.trustworthy6g.net/tdp/sentra/sentra-no-sgx:latest "$@"
