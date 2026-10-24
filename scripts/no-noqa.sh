#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

allowed_marks=(
  "cuml-test/cuml/tests/load-test.rkt:(require (only-in cuml) ;; noqa"
  "cuml-test/cuml/tests/private/python-env.rkt:(define twin-skip ;; noqa"
  "lint/review.rkt:(define (review-scoped-body leading body) ;; noqa"
)
facades=(cuml-lib/cuml/main.rkt cuml-core-lib/cuml/core.rkt cuml-cluster-lib/cuml/cluster.rkt)

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

hits=$(grep -rnE --include='*.rkt' --include='*.scrbl' \
  ';; (noqa|lint: ignore|review: ignore)$|#\|review: ignore\|#' .fmt.rkt cuml-*/ lint scripts || true)

status=0
while IFS=: read -r file line text; do
  [ -n "$file" ] || continue
  allowed "$file" "$text" && continue
  status=1
  message="lint suppression: teach lint/review.rkt the form instead"
  echo "$file:$line: $message"
  if [ -n "${GITHUB_ACTIONS:-}" ]; then
    echo "::error file=$file,line=$line::$message"
  fi
done <<<"$hits"

if [ "$status" -eq 0 ]; then
  echo "no-noqa: no new lint suppression in the cuml packages, lint or scripts"
fi
exit "$status"
