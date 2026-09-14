# Run from benchmarking/ml-benchmark/ (Compose context = repo root).

COMPOSE_FILE := $(dir $(lastword $(MAKEFILE_LIST)))docker-compose.yml
COMPOSE := docker compose -f "$(COMPOSE_FILE)"

.PHONY: help build-ml build-ml-sgx run-quick run-standard run-with-client run-with-client-2epoch run-with-client-versioned run-with-client-metrics-3node run-with-client-fault-5node compare-versioning shell-ml shell-ml-sgx build-sentra-sgx metrics-collect sgx-run-with-client-2epoch sgx-run-with-client-metrics-3node sgx-run-with-client-fault-5node

help:
	@echo "benchmarking/ml-benchmark — see README.md"
	@echo "  make build-ml          - sentra-ml-benchmark:latest"
	@echo "  make build-ml-sgx          - sentra-ml-benchmark-sgx:latest"
	@echo "  make run-quick|run-standard|run-with-client|run-with-client-versioned"
	@echo "  make compare-versioning BASELINE_RUN=... VERSIONED_RUN=..."
	@echo "  make metrics-collect RUN_DIR=..."
	@echo "  make sgx-run-quick|sgx-run-standard|sgx-run-with-client|sgx-run-with-client-2epoch|sgx-run-with-client-fault-5node"
	@echo "  make shell-ml"
	@echo "  make shell-ml-sgx"
	@echo "  make build-sentra-sgx  - ../sentra-node/ Occlum image (long build)"

build-ml:
	$(COMPOSE) build ml-benchmark

build-ml-sgx:
	$(COMPOSE) build ml-benchmark-sgx

run-quick:
	$(COMPOSE) run --rm ml-benchmark --config /workspace/benchmark-configs/quick.yaml

run-standard:
	$(COMPOSE) run --rm ml-benchmark --config /workspace/benchmark-configs/standard.yaml

run-with-client:
	$(COMPOSE) run --rm ml-benchmark --config /workspace/benchmark-configs/with-client.yaml

run-with-client-versioned:
	$(COMPOSE) run --rm ml-benchmark --config /workspace/benchmark-configs/with-client-versioned.yaml

run-with-client-2epoch:
	$(COMPOSE) run --rm ml-benchmark --config /workspace/benchmark-configs/with-client-2epoch.yaml

run-with-client-metrics-3node:
	$(COMPOSE) run --rm ml-benchmark --config /workspace/benchmark-configs/with-client-metrics-3node.yaml

run-with-client-fault-5node:
	$(COMPOSE) run --rm ml-benchmark --config /workspace/benchmark-configs/with-client-fault-5node.yaml

BASELINE_RUN ?=
VERSIONED_RUN ?=
compare-versioning:
	@if [ -z "$(BASELINE_RUN)" ] || [ -z "$(VERSIONED_RUN)" ]; then \
		echo "Usage: make compare-versioning BASELINE_RUN=/path/to/run_baseline VERSIONED_RUN=/path/to/run_versioned"; \
		exit 1; fi
	$(COMPOSE) run --rm --entrypoint python3 ml-benchmark \
		/usr/local/bin/compare_versioning_overhead.py \
		"$(BASELINE_RUN)" "$(VERSIONED_RUN)"

sgx-run-quick:
	$(COMPOSE) run --rm ml-benchmark-sgx sgx --config /workspace/benchmark-configs/quick.yaml

sgx-run-standard:
	$(COMPOSE) run --rm ml-benchmark-sgx sgx  --config /workspace/benchmark-configs/standard.yaml

sgx-run-with-client:
	$(COMPOSE) run --rm ml-benchmark-sgx sgx --config /workspace/benchmark-configs/with-client.yaml

sgx-run-with-client-2epoch:
	$(COMPOSE) run --rm ml-benchmark-sgx sgx --config /workspace/benchmark-configs/with-client-2epoch.yaml

sgx-run-with-client-metrics-3node:
	$(COMPOSE) run --rm ml-benchmark-sgx sgx --config /workspace/benchmark-configs/with-client-metrics-3node.yaml

sgx-run-with-client-fault-5node:
	$(COMPOSE) run --rm ml-benchmark-sgx sgx --config /workspace/benchmark-configs/with-client-fault-5node.yaml

shell-ml:
	$(COMPOSE) run --rm ml-benchmark bash

shell-ml-sgx:
	$(COMPOSE) run --rm ml-benchmark-sgx bash

build-sentra-sgx:
	cd "$(dir $(lastword $(MAKEFILE_LIST)))../sentra-node" && sh ./buildSentraNodeDockerImage.sh

RUN_DIR ?=
metrics-collect:
	@if [ -z "$(RUN_DIR)" ]; then echo "Set RUN_DIR=path/to/run_YYYYMMDD_HHMMSS"; exit 1; fi
	$(COMPOSE) run --rm --entrypoint python3 ml-benchmark \
		/usr/local/bin/collect_run_metrics.py "$(RUN_DIR)"
