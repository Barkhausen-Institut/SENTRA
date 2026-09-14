#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec docker build -f Dockerfile.sentra-node-sgx -t registry.tdp.trustworthy6g.net/tdp/sentra/sentra-sgx:latest ../..
