#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade build pybind11 wheel
python -m build --wheel
