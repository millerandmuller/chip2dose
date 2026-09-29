# One command from a fresh clone: `make` (needs Python 3.11 on PATH; override with PYTHON=...)
PYTHON ?= python3.11
VENV   := .venv
PY     := $(VENV)/bin/python

all: results

$(PY): requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements.txt
	touch $(PY)

data: $(PY)
	$(PY) data/fetch.py

results: data
	$(PY) run_demo.py

quick: data
	$(PY) run_demo.py --skip-validation

test: data
	$(PY) -m pytest -q

clean:
	rm -rf results

.PHONY: all data results quick test clean
