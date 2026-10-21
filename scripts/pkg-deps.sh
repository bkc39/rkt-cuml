#!/usr/bin/env bash
#
# raco setup's dependency check over the given packages, red on an unused
# dependency too: setup prints that one to stderr and still exits 0.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ "$#" -gt 0 ] || { echo "usage: scripts/pkg-deps.sh <pkg> ..." >&2; exit 2; }
log=$(mktemp)
trap 'rm -f "$log"' EXIT

raco setup -j 4 --no-docs --check-pkg-deps --unused-pkg-deps --pkgs "$@" 2>&1 | tee "$log"

if grep -qE "unused dependenc(y|ies) detected" "$log"; then
  echo "pkg-deps: unused dependency (above)" >&2
  exit 1
fi
echo "pkg-deps: no missing or unused dependency in $*"
