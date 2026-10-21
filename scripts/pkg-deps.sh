#!/usr/bin/env bash
#
# raco setup's dependency check over the given packages, red on an unused
# dependency too: setup prints that one to stderr and still exits 0. With
# --no-docs, setup cannot see the packages a manual links into, so those are
# listed in doc_only.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

doc_only=(cuml-doc:racket-doc cuml-doc:threading-doc)

[ "$#" -gt 0 ] || { echo "usage: scripts/pkg-deps.sh <pkg> ..." >&2; exit 2; }
log=$(mktemp)
trap 'rm -f "$log"' EXIT

raco setup -j 4 --no-docs --check-pkg-deps --unused-pkg-deps --pkgs "$@" 2>&1 | tee "$log"

mapfile -t unused < <(awk '
  /unused dependenc(y|ies) detected/ { block = 1; next }
  block && /^  for package: / { pkg = $3; gsub(/"/, "", pkg); next }
  block && /^  on package/ { next }
  block && /^   "/ { dep = $1; gsub(/"/, "", dep); print pkg ":" dep; next }
  { block = 0 }' "$log")

status=0
for entry in "${unused[@]}"; do
  [[ " ${doc_only[*]} " == *" $entry "* ]] && continue
  echo "pkg-deps: unused dependency ${entry#*:} of ${entry%%:*}" >&2
  status=1
done
[ "$status" -eq 0 ] && echo "pkg-deps: no missing or unused dependency in $*"
exit "$status"
