# SENTRA deployment manifests

This directory contains the backend-assisted deployment manifests. It is
separate from the self-contained MPC benchmark workflow under
`benchmarking/ml-benchmark/`.

## Important backend dependency

The Compose and Kubernetes manifests require:

```text
registry.tdp.trustworthy6g.net/tdp/sentra/sentra-backend:latest
```

The backend source and local Dockerfile are not included in this repository.
You must have permission to pull that private image (or replace it with a
compatible image) before using these manifests. The node images still build
locally from `benchmarking/sentra-node/`.

## Files

- `docker-compose.yaml` — SGX and NoSGX node profiles plus the external backend.
- `sentra-kubernetes.yaml` — SGX Kubernetes deployment.
- `sentra-kubernetes-no-sgx.yaml` — NoSGX Kubernetes deployment.

The obsolete Swarm example was removed because Docker Swarm does not build
images from a Compose `build` section and its referenced Dockerfile no longer
exists.

## Requirements

- Linux host with a recent Docker Engine and Compose plugin.
- Intel SGX devices and Occlum prerequisites for SGX profiles.
- Access to the private backend image listed above.
- Repository root as the Docker build context.

## Compose profiles

From this directory:

```bash
docker login registry.tdp.trustworthy6g.net
docker compose --profile <profile> up
```

Available profiles:

- `test` — simple training using the SGX node image.
- `test-no-sgx` — simple training using the NoSGX node image.
- `mnist_batched_secure` — secure MNIST using the SGX node image.
- `mnist_batched_secure-no-sgx` — secure MNIST using the NoSGX node image.
- `multi-node` — five SGX node workers.

For secure training without the external backend, use the local Python or
`benchmarking/ml-benchmark/` instructions in the root README.
