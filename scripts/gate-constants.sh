#!/usr/bin/env bash
# gate-constants.sh [header] [module]: the MBSGD gate constants the Racket tests
# restate as literals must equal the shim's.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

header=${1:-shim/linear_model/src/mbsgd_checks.hpp}
module=${2:-cuml-test/cuml/tests/private/mbsgd-gate-data.rkt}
names=(round_off_coef_steps round_off_intercept_share)

status=0
for name in "${names[@]}"; do
  racket_name=${name//_/-}
  shim=$(sed -nE "s/^inline constexpr double $name = ([0-9.e+-]+);$/\1/p" "$header")
  literal=$(sed -nE "s/^\(define $racket_name ([0-9.e+-]+)\)$/\1/p" "$module")
  if [ -z "$shim" ] || [ -z "$literal" ]; then
    message="$name: not found as a constexpr in $header and a define in $module"
    status=1
  elif ! awk -v a="$shim" -v b="$literal" 'BEGIN { exit !(a + 0 == b + 0) }'; then
    message="$name: $header has $shim, $module has $literal"
    status=1
  else
    continue
  fi
  echo "$module: $message"
  if [ -n "${GITHUB_ACTIONS:-}" ]; then
    echo "::error file=$module::$message"
  fi
done

if [ "$status" -eq 0 ]; then
  echo "gate-constants: ${names[*]} match between the shim and the tests"
fi
exit "$status"
