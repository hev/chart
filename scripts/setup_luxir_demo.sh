#!/usr/bin/env bash
# Linux local lexical demo only: no embedding packages, models or credentials.
set -euo pipefail
artifact_root=${1:?usage: setup_luxir_demo.sh ARTIFACT_ROOT GATEWAY_CHECKOUT}
gateway_checkout=${2:?usage: setup_luxir_demo.sh ARTIFACT_ROOT GATEWAY_CHECKOUT}
script_dir=$(cd "$(dirname "$0")" && pwd)
test -f "$gateway_checkout/clients/python/pyproject.toml"
python3 -c 'import sys; assert sys.version_info >= (3, 11), "Python >=3.11 required"'
mkdir -p "$artifact_root"
python3 -m venv "$artifact_root/venv"
timeout 180 "$artifact_root/venv/bin/pip" install -r "$script_dir/luxir-demo-requirements.txt" -e "$gateway_checkout/clients/python"
echo "Chart lexical demo Python: $artifact_root/venv/bin/python"
