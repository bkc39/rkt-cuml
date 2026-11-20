#!/usr/bin/env bash
# no-twin-copies.sh [root]: twin_common.py's host and output, and the Python
# pins, stay in their homes under cuml-test.
set -euo pipefail

cd "${1:-$(dirname "${BASH_SOURCE[0]}")/..}"

common=cuml-test/cuml/tests/python/twin_common.py
pins=cuml-test/cuml/tests/private/python-env.rkt

status=0

refuse() {
  local hits=$1 home=$2 message=$3 file line
  while IFS=: read -r file line _; do
    [ -n "$file" ] || continue
    [ "$file" = "$home" ] && continue
    status=1
    echo "$file:$line: $message"
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
      echo "::error file=$file,line=$line::$message"
    fi
  done <<<"$hits"
}

python() {
  grep -rnF --include='*.py' "$1" cuml-test || true
}

refuse "$(python 'def host(')" "$common" \
  "def host( outside $common: import host from twin_common"
refuse "$(python 'json.dump(')" "$common" \
  "json.dump( outside $common: write the twin's output with twin_common.run"
refuse "$(grep -rnF --exclude-dir=compiled -e '"26.08.00"' -e '"1.8.0"' cuml-test || true)" "$pins" \
  "a pinned Python version outside $pins: use python-cuml-version or python-sklearn-version"

if [ "$status" -eq 0 ]; then
  echo "no-twin-copies: host and json.dump only in $common, the Python pins only in $pins"
fi
exit "$status"
