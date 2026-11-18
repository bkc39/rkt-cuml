#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ $# -gt 0 ]; then
  dirs=("$@")
else
  dirs=(cuml-test)
fi
before='(^|[^-[:alnum:]!?*<>=/+.$%&^~_])'
after='([^-[:alnum:]!?*<>=/+:.$%&^~_]|$)'
message="read a test table's rows with match-define, not caddr or cadddr"

hits=$(grep -rnE --include='*.rkt' "${before}cadd?dr${after}" "${dirs[@]}" || true)

status=0
while IFS=: read -r file line _; do
  [ -n "$file" ] || continue
  status=1
  echo "$file:$line: $message"
  if [ -n "${GITHUB_ACTIONS:-}" ]; then
    echo "::error file=$file,line=$line::$message"
  fi
done <<<"$hits"

if [ "$status" -eq 0 ]; then
  echo "no-positional-access: no caddr or cadddr in ${dirs[*]}"
fi
exit "$status"
