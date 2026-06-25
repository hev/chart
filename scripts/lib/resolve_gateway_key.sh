resolve_gateway_key() {
  if [[ -n "${LAYER_GATEWAY_API_KEY:-}" ]]; then
    export LAYER_GATEWAY_API_KEY
    return 0
  fi

  if [[ -f .env ]]; then
    local env_line env_value
    env_line="$(grep -E '^[[:space:]]*LAYER_GATEWAY_API_KEY=' .env | tail -n 1 || true)"
    env_value="${env_line#*=}"
    env_value="${env_value%%#*}"
    env_value="${env_value#"${env_value%%[![:space:]]*}"}"
    env_value="${env_value%"${env_value##*[![:space:]]}"}"
    env_value="${env_value%\"}"
    env_value="${env_value#\"}"
    env_value="${env_value%\'}"
    env_value="${env_value#\'}"
    if [[ -n "$env_value" ]]; then
      LAYER_GATEWAY_API_KEY="$env_value"
      export LAYER_GATEWAY_API_KEY
      return 0
    fi
  fi

  if command -v op >/dev/null 2>&1; then
    local op_error op_output op_pid waited timeout_seconds op_refs op_ref op_items op_item op_vault op_field
    op_error="$(mktemp)"
    op_output="$(mktemp)"
    timeout_seconds="${CHART_OP_TIMEOUT_SECONDS:-15}"
    op_vault="${CHART_GATEWAY_KEY_OP_VAULT:-mesh-staging}"
    op_field="${CHART_GATEWAY_KEY_OP_FIELD:-credential}"
    if [[ -n "${CHART_GATEWAY_KEY_OP_ITEM:-}" ]]; then
      op_items=("$CHART_GATEWAY_KEY_OP_ITEM")
    else
      op_items=("layer turbopuffer" "layer-turbopuffer")
    fi
    if [[ -n "${CHART_GATEWAY_KEY_OP_REF:-}" ]]; then
      op_refs=("$CHART_GATEWAY_KEY_OP_REF")
    else
      op_refs=(
        "op://mesh-staging/layer turbopuffer/credential"
        "op://mesh-staging/layer-turbopuffer/credential"
      )
    fi
    if command -v python3 >/dev/null 2>&1; then
      if python3 - "$timeout_seconds" "$op_output" "$op_error" "$op_vault" "$op_field" "${op_items[@]}" -- "${op_refs[@]}" <<'PY'
import subprocess
import sys
import os
import signal

timeout = float(sys.argv[1])
out_path = sys.argv[2]
err_path = sys.argv[3]
vault = sys.argv[4]
field = sys.argv[5]
parts = sys.argv[6:]
sep = parts.index("--")
items = parts[:sep]
refs = parts[sep + 1 :]
errors = []
timed_out = False
commands = [
    (f"{vault}/{item}/{field}", ["op", "item", "get", item, "--vault", vault, "--field", field, "--reveal"])
    for item in items
] + [(ref, ["op", "read", ref]) for ref in refs]
for label, cmd in commands:
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        timed_out = True
        errors.append(f"{label}: timed out")
        continue
    if proc.returncode == 0:
        open(out_path, "w").write(stdout)
        open(err_path, "w").write("")
        sys.exit(0)
    errors.append(f"{label}: {stderr.strip()}")

open(err_path, "w").write("; ".join(errors))
if timed_out:
    sys.exit(124)
sys.exit(1)
PY
      then
        LAYER_GATEWAY_API_KEY="$(<"$op_output")"
        rm -f "$op_error" "$op_output"
        export LAYER_GATEWAY_API_KEY
        return 0
      else
        local op_status="$?"
        if [[ "$op_status" == "124" ]]; then
          echo "Timed out reading gateway key from 1Password after ${timeout_seconds}s per attempt." >&2
        else
          echo "Could not read gateway key from 1Password: $(<"$op_error")" >&2
        fi
        rm -f "$op_error" "$op_output"
      fi
    else
      for op_item in "${op_items[@]}"; do
        if op item get "$op_item" --vault "$op_vault" --field "$op_field" --reveal >"$op_output" 2>"$op_error"; then
          LAYER_GATEWAY_API_KEY="$(<"$op_output")"
          rm -f "$op_error" "$op_output"
          export LAYER_GATEWAY_API_KEY
          return 0
        fi
      done
      for op_ref in "${op_refs[@]}"; do
        op read "$op_ref" >"$op_output" 2>"$op_error" &
        op_pid="$!"
        waited=0
        while kill -0 "$op_pid" >/dev/null 2>&1 && [[ "$waited" -lt "$timeout_seconds" ]]; do
          sleep 1
          waited=$((waited + 1))
        done
        if kill -0 "$op_pid" >/dev/null 2>&1; then
          kill "$op_pid" >/dev/null 2>&1 || true
          sleep 0.2
          if kill -0 "$op_pid" >/dev/null 2>&1; then
            kill -9 "$op_pid" >/dev/null 2>&1 || true
          fi
          wait "$op_pid" >/dev/null 2>&1 || true
          echo "Timed out reading gateway key from 1Password after ${timeout_seconds}s per attempt." >&2
          break
        elif wait "$op_pid"; then
          LAYER_GATEWAY_API_KEY="$(<"$op_output")"
          rm -f "$op_error" "$op_output"
          export LAYER_GATEWAY_API_KEY
          return 0
        fi
      done
      echo "Could not read gateway key from 1Password: $(<"$op_error")" >&2
      rm -f "$op_error" "$op_output"
    fi
  fi

  echo "LAYER_GATEWAY_API_KEY is required. Export it first or sign in with: op signin" >&2
  return 1
}
