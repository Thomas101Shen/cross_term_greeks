#!/usr/bin/env bash
set -euo pipefail

PYTHON_TAG="${PYTHON_TAG:-cp310-cp310}"
IMAGE="${MANYLINUX_IMAGE:-quay.io/pypa/manylinux2014_x86_64}"

docker run --rm \
  -v "$(pwd)":/io \
  -w /io \
  "${IMAGE}" \
  /bin/bash -lc "
    set -euo pipefail
    /opt/python/${PYTHON_TAG}/bin/python -m pip install --upgrade build pybind11 wheel auditwheel
    /opt/python/${PYTHON_TAG}/bin/python -m build --wheel
    auditwheel repair dist/*.whl -w wheelhouse
  "
