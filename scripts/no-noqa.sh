#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/grep-gate.sh
source "$(dirname "${BASH_SOURCE[0]}")/grep-gate.sh"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

allowed_marks=(
  "cuml-test/cuml/tests/load-test.rkt:(require (only-in cuml) ;; noqa"
  "cuml-test/cuml/tests/private/python-env.rkt:(define twin-skip ;; noqa"
  "lint/review.rkt:(define (review-scoped-body leading body) ;; noqa"
)
facades=(cuml-lib/cuml/main.rkt cuml-cluster-lib/cuml/cluster.rkt)

# shellcheck disable=SC2329
allowed() {
  local file=$1 text=$2 entry
  if [ "$text" = '#|review: ignore|#' ]; then
    [ "$(basename "$file")" = info.rkt ] && return 0
    for entry in "${facades[@]}"; do
      [ "$file" = "$entry" ] && return 0
    done
    return 1
  fi
  for entry in "${allowed_marks[@]}"; do
    [ "$file:$text" = "$entry" ] && return 0
  done
  return 1
}

status=0
found=$(hits -rnE --include='*.rkt' --include='*.scrbl' \
  ';; (noqa|lint: ignore|review: ignore)$|#\|review: ignore\|#' .fmt.rkt cuml-*/ lint scripts)
refuse "$found" "lint suppression: teach lint/review.rkt the form instead" allowed || status=1

if [ "$status" -eq 0 ]; then
  echo "no-noqa: no new lint suppression in the cuml packages, lint or scripts"
fi
exit "$status"
