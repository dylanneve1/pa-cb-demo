#!/bin/bash
# PA CB demo runner (Linux). Same flow as run_demo.ps1: point it at a python
# env that has the PA build's openvino + openvino_genai wheels installed, and
# it tees the demo and parity runs into logs/.
#
# Usage: MODEL_DIR=/path/to/model ./run_demo.sh [/path/to/venv]
set -e

here="$(cd "$(dirname "$0")" && pwd)"
venv="${1:-}"
[ -n "$venv" ] && . "$venv/bin/activate"

: "${MODEL_DIR:?set MODEL_DIR to the int4 stateful model export}"
export MODEL_DIR

mkdir -p "$here/logs"
stamp=$(date +%Y%m%d-%H%M%S)

echo "=== demo (quiet) ==="
python3 "$here/pa_demo.py" cb llm 2>&1 | tee "$here/logs/${stamp}_01_demo.log"

echo "=== demo (NPUW log INFO, shows the PA front-end engaging) ==="
python3 "$here/pa_demo.py" cb llm --npuw-log INFO 2>&1 | tee "$here/logs/${stamp}_02_demo_npuw_info.log"

echo "=== parity + similarity: plain CPU vs NPU + NPUW_PA ==="
python3 "$here/pa_demo.py" parity 2>&1 | tee "$here/logs/${stamp}_03_parity.log"

echo "All runs green. Logs in $here/logs"
