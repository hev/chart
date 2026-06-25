write_failure_report() {
  local report_path="$1"
  local status="$2"
  local error="$3"
  local report_tmp="${report_path}.tmp.$$"
  mkdir -p "$(dirname "$report_path")"
  python3 - "$status" "$error" >"$report_tmp" <<'PY'
import json
import sys

status, error = sys.argv[1:3]
print(json.dumps({"status": status, "error": error}, indent=2))
PY
  mv "$report_tmp" "$report_path"
}
