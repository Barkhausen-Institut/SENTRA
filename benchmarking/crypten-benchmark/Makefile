.PHONY: help install run

CONFIG ?= configs/sentra-with-client.yaml
PYTHON ?= python3

help:
	@echo "CrypTen vs SENTRA comparison benchmark"
	@echo "  make install   # pip install deps + crypten (sklearn workaround)"
	@echo "  make run       # launch MPC parties (default 3)"

# crypten 0.4.1 depends on deprecated PyPI package "sklearn"; pip blocks it unless
# scikit-learn is present and SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=1.
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-crypten.txt
	SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
		$(PYTHON) -m pip install crypten==0.4.1

run:
	$(PYTHON) launcher.py -c $(CONFIG)
