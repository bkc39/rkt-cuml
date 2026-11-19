#!/usr/bin/env bash
# gate-constants.sh [header] [module] [rows-header] [rows-module]: the MBSGD
# gate constants the Racket tests restate as literals, and the row minimum
# labels.rkt restates, must equal the shim's.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

header=${1:-shim/linear_model/src/mbsgd_checks.hpp}
module=${2:-cuml-test/cuml/tests/private/mbsgd-gate-data.rkt}
names=(gate_floor_share gate_round_off_factor round_off_coef_steps
  round_off_intercept_share round_off_prediction_share)
per_dtype=(gate_round_off_start)
rows_header=${3:-shim/linear_model/src/min_rows.hpp}
rows_module=${4:-cuml-linear-model-lib/cuml/linear-model/private/labels.rkt}

compare() {
  local name=$1 shim=$2 literal=$3 header=${4:-$header} module=${5:-$module} message
  if [ -z "$shim" ] || [ -z "$literal" ]; then
    message="$name: not found as a constexpr in $header and a define in $module"
  elif ! awk -v a="$shim" -v b="$literal" 'BEGIN { exit !(a + 0 == b + 0) }'; then
    message="$name: $header has $shim, $module has $literal"
  else
    return 0
  fi
  echo "$module: $message"
  if [ -n "${GITHUB_ACTIONS:-}" ]; then
    echo "::error file=$module::$message"
  fi
  return 1
}

status=0
for name in "${names[@]}"; do
  shim=$(sed -nE "s/^inline constexpr double $name = ([0-9.e+-]+);$/\1/p" "$header")
  literal=$(sed -nE "s/^\(define ${name//_/-} ([0-9.e+-]+)\)$/\1/p" "$module")
  compare "$name" "$shim" "$literal" || status=1
done
for name in "${per_dtype[@]}"; do
  shim=$(sed -nE "/^inline constexpr double $name =$/{n;s/^ *std::is_same_v<T, float> \? ([0-9.e+-]+) : ([0-9.e+-]+);$/\1 \2/p}" "$header")
  literal=$(sed -nE "s/^\(define ${name//_/-} '\(\(float32 \. ([0-9.e+-]+)\) \(float64 \. ([0-9.e+-]+)\)\)\)$/\1 \2/p" "$module")
  compare "$name float32" "${shim% *}" "${literal% *}" || status=1
  compare "$name float64" "${shim#* }" "${literal#* }" || status=1
done
shim=$(sed -nE "s/^inline constexpr int64_t min_rows = ([0-9]+);$/\1/p" "$rows_header")
literal=$(sed -nE "s/^\(define shim-min-rows ([0-9]+)\)$/\1/p" "$rows_module")
compare min_rows "$shim" "$literal" "$rows_header" "$rows_module" || status=1

if [ "$status" -eq 0 ]; then
  echo "gate-constants: ${names[*]} ${per_dtype[*]} min_rows match between the shim and the Racket"
fi
exit "$status"
