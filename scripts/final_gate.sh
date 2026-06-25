#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for arg in "$@"; do
  if [[ "$arg" == "--ready" ]]; then
    echo "scripts/final_gate.sh is the hard completion gate; use scripts/plan_audit.sh --ready for runnable next steps." >&2
    exit 2
  fi
done

scripts/plan_audit.sh --requirements --require-complete "$@"
