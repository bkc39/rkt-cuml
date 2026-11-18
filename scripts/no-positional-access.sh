#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ $# -gt 0 ]; then
  dirs=("$@")
else
  dirs=(cuml-test cuml-glmnet-test)
fi
refused=cadr,caddr,cadddr,third,fourth

status=0
hits=$(racket lint/positional.rkt "$refused" "${dirs[@]}") || status=$?
if [ "$status" -gt 1 ]; then
  echo "$hits"
  exit "$status"
fi
while IFS= read -r hit; do
  [ -n "$hit" ] || continue
  echo "$hit"
  if [ -n "${GITHUB_ACTIONS:-}" ]; then
    IFS=: read -r file line _ <<<"$hit"
    echo "::error file=$file,line=$line::${hit#*: *: }"
  fi
done <<<"$hits"

if [ "$status" -eq 0 ]; then
  echo "no-positional-access: no ${refused//,/, } in ${dirs[*]}"
fi
exit "$status"
