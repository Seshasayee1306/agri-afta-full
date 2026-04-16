#!/usr/bin/env bash
set -u

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
FAILS=0
CASE_FILE=""
CASE_STATUS=""

print_json() {
  local file="$1"
  python3 - "$file" <<'PY'
import json
import sys

path = sys.argv[1]
text = open(path, "r", encoding="utf-8").read()
try:
    obj = json.loads(text)
    print(json.dumps(obj, indent=2, ensure_ascii=True))
except Exception:
    print(text)
PY
}

post_case() {
  local name="$1"
  local endpoint="$2"
  local payload="$3"

  CASE_FILE="$(mktemp)"
  CASE_STATUS="$(
    curl -sS -o "$CASE_FILE" -w "%{http_code}" \
      -X POST "${BASE_URL}${endpoint}" \
      -H "Content-Type: application/json" \
      -d "$payload"
  )"

  echo
  echo "============================================================"
  echo "CASE: ${name}"
  echo "POST: ${BASE_URL}${endpoint}"
  echo "HTTP: ${CASE_STATUS}"
  echo "BODY:"
  print_json "$CASE_FILE"
}

assert_status() {
  local expected="$1"
  if [[ "$CASE_STATUS" == "$expected" ]]; then
    echo "[PASS] HTTP status is ${expected}"
  else
    echo "[FAIL] Expected HTTP ${expected}, got ${CASE_STATUS}"
    FAILS=$((FAILS + 1))
  fi
}

assert_expr() {
  local expr="$1"
  local message="$2"
  if python3 - "$CASE_FILE" "$expr" <<'PY'
import json
import sys

path = sys.argv[1]
expr = sys.argv[2]
obj = json.load(open(path, "r", encoding="utf-8"))
ok = bool(eval(expr, {}, {"obj": obj}))
raise SystemExit(0 if ok else 1)
PY
  then
    echo "[PASS] ${message}"
  else
    echo "[FAIL] ${message}"
    FAILS=$((FAILS + 1))
  fi
}

cleanup_case() {
  if [[ -n "$CASE_FILE" && -f "$CASE_FILE" ]]; then
    rm -f "$CASE_FILE"
  fi
  CASE_FILE=""
  CASE_STATUS=""
}

run() {
  local payload_valid='{
    "sowing_date": "2026-04-10",
    "current_date": "2026-04-16",
    "soil_moisture": 38,
    "temperature": 31,
    "humidity": 42,
    "ph": 6.4,
    "region": "south india",
    "crop_type": "rice",
    "soil_type": "clay",
    "soil_humidity": 42,
    "rainfall": 77.8,
    "air_temp": 31,
    "air_humidity": 42,
    "nitrogen": 37,
    "phosphorus": 51,
    "potassium": 32,
    "challenger_kind": "catboost"
  }'

  # 1) Happy path
  post_case "1) Happy path valid payload" "/predict_catboost_compare" "$payload_valid"
  assert_status "200"
  assert_expr 'obj.get("model_family") == "catboost"' 'model_family is catboost'
  assert_expr 'obj.get("final_prediction") in (0, 1)' 'final_prediction exists'
  cleanup_case

  # 2) CatBoost lock check without challenger_kind
  post_case "2) CatBoost lock check (no challenger_kind)" "/predict_catboost_compare" '{
    "sowing_date": "2026-04-10",
    "current_date": "2026-04-16",
    "soil_moisture": 38,
    "temperature": 31,
    "humidity": 42,
    "ph": 6.4,
    "region": "south india",
    "crop_type": "rice",
    "soil_type": "clay",
    "soil_humidity": 42,
    "rainfall": 77.8,
    "air_temp": 31,
    "air_humidity": 42,
    "nitrogen": 37,
    "phosphorus": 51,
    "potassium": 32
  }'
  assert_status "200"
  assert_expr 'obj.get("model_family") == "catboost"' 'route still serves catboost'
  cleanup_case

  # 3) Missing required field
  post_case "3) Missing required field (sowing_date)" "/predict_catboost_compare" '{
    "current_date": "2026-04-16",
    "soil_moisture": 38,
    "temperature": 31,
    "humidity": 42,
    "ph": 6.4,
    "region": "south india",
    "crop_type": "rice",
    "soil_type": "clay"
  }'
  assert_status "400"
  assert_expr '"error" in obj' 'error message returned'
  cleanup_case

  # 4a) Boundary valid payload (minimums)
  post_case "4a) Boundary valid mins" "/predict_catboost_compare" '{
    "sowing_date": "2026-01-01",
    "current_date": "2026-01-01",
    "soil_moisture": 0,
    "temperature": -20,
    "humidity": 0,
    "ph": 0,
    "region": "south india",
    "crop_type": "rice",
    "soil_type": "clay",
    "soil_humidity": 0,
    "rainfall": 0,
    "air_temp": -20,
    "air_humidity": 0,
    "nitrogen": 0,
    "phosphorus": 0,
    "potassium": 0
  }'
  assert_status "200"
  assert_expr 'obj.get("final_prediction") == 1' 'safety path can force irrigation at extreme dry mins'
  cleanup_case

  # 4b) Out-of-range payload
  post_case "4b) Out-of-range value" "/predict_catboost_compare" '{
    "sowing_date": "2026-04-10",
    "current_date": "2026-04-16",
    "soil_moisture": 38,
    "temperature": 31,
    "humidity": 42,
    "ph": 14.5,
    "region": "south india",
    "crop_type": "rice",
    "soil_type": "clay"
  }'
  assert_status "400"
  assert_expr '"error" in obj' 'range validation returns error'
  cleanup_case

  # 5) Safety override: soil_humidity = 0
  post_case "5) Safety override soil_humidity=0" "/predict_catboost_compare" '{
    "sowing_date": "2026-04-10",
    "current_date": "2026-04-16",
    "soil_moisture": 38,
    "temperature": 31,
    "humidity": 42,
    "ph": 6.4,
    "region": "south india",
    "crop_type": "rice",
    "soil_type": "clay",
    "soil_humidity": 0
  }'
  assert_status "200"
  assert_expr 'obj.get("final_prediction") == 1' 'forced irrigation when soil_humidity is zero'
  assert_expr 'obj.get("safety_override", {}).get("applied") is True' 'safety_override.applied is true'
  cleanup_case

  # 6) Safety override: soil_moisture = 0
  post_case "6) Safety override soil_moisture=0" "/predict_catboost_compare" '{
    "sowing_date": "2026-04-10",
    "current_date": "2026-04-16",
    "soil_moisture": 0,
    "temperature": 31,
    "humidity": 42,
    "ph": 6.4,
    "region": "south india",
    "crop_type": "rice",
    "soil_type": "clay",
    "soil_humidity": 42
  }'
  assert_status "200"
  assert_expr 'obj.get("final_prediction") == 1' 'forced irrigation when soil_moisture is zero'
  assert_expr 'obj.get("safety_override", {}).get("applied") is True' 'safety_override.applied is true'
  cleanup_case

  # 7) Safety override: critical heat
  post_case "7) Safety override critical heat" "/predict_catboost_compare" '{
    "sowing_date": "2026-04-10",
    "current_date": "2026-04-16",
    "soil_moisture": 38,
    "temperature": 46,
    "humidity": 42,
    "ph": 6.4,
    "region": "south india",
    "crop_type": "rice",
    "soil_type": "clay",
    "soil_humidity": 42,
    "air_temp": 46
  }'
  assert_status "200"
  assert_expr 'obj.get("final_prediction") == 1' 'forced irrigation on critical heat'
  assert_expr 'obj.get("safety_override", {}).get("applied") is True' 'safety_override.applied is true'
  cleanup_case

  # 8) Safety override: dry combo
  post_case "8) Safety override dry combo" "/predict_catboost_compare" '{
    "sowing_date": "2026-04-10",
    "current_date": "2026-04-16",
    "soil_moisture": 8,
    "temperature": 31,
    "humidity": 42,
    "ph": 6.4,
    "region": "south india",
    "crop_type": "rice",
    "soil_type": "clay",
    "soil_humidity": 7,
    "rainfall": 0
  }'
  assert_status "200"
  assert_expr 'obj.get("final_prediction") == 1' 'forced irrigation on dry combo'
  assert_expr 'obj.get("safety_override", {}).get("applied") is True' 'safety_override.applied is true'
  cleanup_case

  # 9) Advisory warning only (pH extreme)
  post_case "9) Advisory warning only (pH extreme)" "/predict_catboost_compare" '{
    "sowing_date": "2026-04-10",
    "current_date": "2026-04-16",
    "soil_moisture": 38,
    "temperature": 31,
    "humidity": 42,
    "ph": 9.2,
    "region": "south india",
    "crop_type": "rice",
    "soil_type": "clay",
    "soil_humidity": 42
  }'
  assert_status "200"
  assert_expr 'isinstance(obj.get("field_warnings"), list) and len(obj.get("field_warnings")) > 0' 'field_warnings populated'
  assert_expr 'obj.get("safety_override") is None' 'no forced override for advisory-only case'
  cleanup_case

  # 10) Generic challenger endpoint with explicit catboost
  post_case "10) Generic endpoint forced catboost" "/predict_challenger_compare" '{
    "sowing_date": "2026-04-10",
    "current_date": "2026-04-16",
    "soil_moisture": 38,
    "temperature": 31,
    "humidity": 42,
    "ph": 6.4,
    "region": "south india",
    "crop_type": "rice",
    "soil_type": "clay",
    "soil_humidity": 42,
    "challenger_kind": "catboost"
  }'
  assert_status "200"
  assert_expr 'obj.get("model_family") == "catboost"' 'generic route honors challenger_kind=catboost'
  cleanup_case

  echo
  echo "============================================================"
  echo "Manual UI checks (11-12):"
  echo "11) Clear localStorage and open /compare -> should show 'Run Prediction Console once first.'"
  echo "12) Temporarily make backend/challenger_models/catboost_model.pkl unavailable -> compare should show fallback banner."
  echo "============================================================"

  if [[ "$FAILS" -gt 0 ]]; then
    echo "Finished with ${FAILS} failure(s)."
    exit 1
  fi
  echo "All automated CatBoost compare API checks passed."
}

run
