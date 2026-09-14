# Sentra node (Docker)

Packaged **Occlum/SGX** node image and legacy **NoSGX** node image used with `sentra-deployment`.

| File | Role |
|------|------|
| `Dockerfile.sentra-node-sgx` | Occlum build + `occlum-instance.tar.gz` runtime. |
| `Dockerfile.sentra-node-nosgx` | Ubuntu + Rust `sentra_node` + conda Python slice. |
| `sentra-node-sbom.yaml` | Occlum BOM for SGX image. |
| `entrypoint-sentra-node.sh` | Waits for `sentra-backend`, `occlum run` or host script. |
| `enclave_run_script_sentra_node.sh` | Starts `sentra_node`. |
| `training_config.yaml` | Copied into the packaged node image. |
| `sgx_sentra_qcnl.conf` | DCAP / QCNL defaults. |

**ML benchmark** (YAML, full Python tree): `../ml-benchmark/`.

## Build SGX image

From **repository root** (long build):

```bash
docker build -f benchmarking/sentra-node/Dockerfile.sentra-node-sgx -t registry.tdp.trustworthy6g.net/tdp/sentra/sentra-sgx:latest .
```

Or from this directory:

```bash
sh ./buildSentraNodeDockerImage.sh
```

These node images are used by the backend-assisted manifests in
`sentra-deployment/`. Running those manifests also requires access to the
external `sentra-backend` registry image; it is not built by these Dockerfiles.
