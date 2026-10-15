#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

pattern='\((malloc|free)([[:space:]]|\)|$)'

hits=$(grep -rnE --include='*.rkt' --include='*.scrbl' "$pattern" cuml-*/ lint scripts || true)

if [ -n "$hits" ]; then
  while IFS=: read -r file line _; do
    message="raw malloc/free: use raft's array constructors or a with-* form"
    echo "$file:$line: $message"
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
      echo "::error file=$file,line=$line::$message"
    fi
  done <<<"$hits"
  exit 1
fi
echo "no-raw-malloc: none in the cuml packages, lint or scripts"
