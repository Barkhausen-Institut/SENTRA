#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec docker build -f Dockerfile.ml-benchmark -t sentra-ml-benchmark:latest ../..
