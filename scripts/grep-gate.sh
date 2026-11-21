# shellcheck shell=bash

hits() {
  local code=0
  grep "$@" || code=$?
  if [ "$code" -gt 1 ]; then
    echo "grep $*: exit $code" >&2
    return "$code"
  fi
}

refuse() {
  local found=$1 message=$2 refused=0 file line text pattern
  shift 2
  while IFS=: read -r file line text; do
    [ -n "$file" ] || continue
    for pattern in "$@"; do
      [[ "$file:$text" =~ $pattern ]] && continue 2
    done
    refused=1
    echo "$file:$line: $message"
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
      echo "::error file=$file,line=$line::$message"
    fi
  done <<<"$found"
  return "$refused"
}
