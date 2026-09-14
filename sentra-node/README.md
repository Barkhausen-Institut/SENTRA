# Attestation using Occlum

For **containerised Python benchmarks** (YAML, Compose), see **`benchmarking/ml-benchmark/README.md`**.

For **packaged Sentra node** Dockerfiles (Occlum/SGX + NoSGX), see **`benchmarking/sentra-node/README.md`** and the index **`benchmarking/README.md`**.

## Running the default occlum example

Launch occlum container by using:

`docker run -it --device /dev/sgx/enclave --device /dev/sgx/provision --device /dev/sgx_provision occlum/occlum:latest-ubuntu20.04 bash`

Configuration of `/etc/sgx_default_qcnl.conf`:
- Configure PCCS URL (currently running on `10.80.1.3`): `"pccs_url": "https://10.80.1.3:8081/sgx/certification/v4/"`
- set use_secure_cert to false: `"use_secure_cert": false`

Basic DCAP test inside the container:
```bash
cd /root/demos/remote_attestation/dcap
./run_dcap_quote_on_occlum.sh
```

## SGX specific attestation details

The **Quoting Enclave** verifies the reports that have been created to its MRENCVLAVE measurement value and then convers and signs them using a device specific asymmetric key, the Intel EPID key.

## Documentation
[Intel DCAP](https://download.01.org/intel-sgx/latest/dcap-latest/linux/docs/SGX_DCAP_Caching_Service_Design_Guide.pdf)


[Intel SGX](https://download.01.org/intel-sgx/latest/linux-latest/docs/Intel_SGX_Developer_Guide.pdf)
