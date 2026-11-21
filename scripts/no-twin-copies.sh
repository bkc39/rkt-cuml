#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/grep-gate.sh
source "$(dirname "${BASH_SOURCE[0]}")/grep-gate.sh"

cd "${1:-$(dirname "${BASH_SOURCE[0]}")/..}"

if [ ! -d cuml-test ]; then
  echo "no-twin-copies: no cuml-test under $PWD: pass the root of an rkt-cuml checkout"
  exit 2
fi

common=cuml-test/cuml/tests/python/twin_common.py
pins=cuml-test/cuml/tests/private/python-env.rkt

home="^${common//./[.]}:"

grep-py() {
  hits -rnF --include='*.py' "$1" cuml-test
}

status=0
found=$(grep-py 'def host(')
refuse "$found" "def host( outside $common: import host from twin_common" "$home" || status=1
found=$(grep-py 'json.dump(')
refuse "$found" "json.dump( outside $common: write the twin's output with twin_common.run" "$home" || status=1
found=$(grep-py 'sys.argv[2]')
refuse "$found" "sys.argv[2] outside $common: write the twin's output with twin_common.run" "$home" || status=1
found=$(hits -rnF --exclude-dir=compiled -e '"26.08.00"' -e '"1.8.0"' cuml-test)
refuse "$found" "a pinned Python version outside $pins: use python-cuml-version or python-sklearn-version" \
  "^${pins//./[.]}:" || status=1

if [ "$status" -eq 0 ]; then
  echo "no-twin-copies: host, json.dump and sys.argv[2] only in $common, the Python pins only in $pins"
fi
exit "$status"
