#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/grep-gate.sh
source "$(dirname "${BASH_SOURCE[0]}")/grep-gate.sh"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

allowed=(
  '^(([^:]*/)?info[.]rkt|cuml-lib/cuml/main[.]rkt|cuml-cluster-lib/cuml/cluster[.]rkt):#[|]review: ignore[|]#$'
  '^cuml-test/cuml/tests/load-test[.]rkt:[(]require [(]only-in cuml[)] ;; noqa$'
  '^cuml-test/cuml/tests/private/python-env[.]rkt:[(]define twin-skip ;; noqa$'
  '^lint/review[.]rkt:[(]define [(]review-scoped-body leading body[)] ;; noqa$'
)

status=0
found=$(hits -rnE --include='*.rkt' --include='*.scrbl' \
  ';; (noqa|lint: ignore|review: ignore)$|#\|review: ignore\|#' .fmt.rkt cuml-*/ lint scripts)
refuse "$found" "lint suppression: teach lint/review.rkt the form instead" "${allowed[@]}" || status=1

if [ "$status" -eq 0 ]; then
  echo "no-noqa: no new lint suppression in the cuml packages, lint or scripts"
fi
exit "$status"
