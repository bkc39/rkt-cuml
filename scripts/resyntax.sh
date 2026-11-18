#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

cache_dir=/var/tmp/rkt-cuml-resyntax-cache
timeout_ms=${RESYNTAX_ANALYZER_TIMEOUT:-120000}
files=()
base_ref=""
while [ $# -gt 0 ]; do
  case $1 in
    --analyzer-timeout)
      timeout_ms=${2:?"resyntax: --analyzer-timeout needs a value in milliseconds"}
      shift 2
      ;;
    --file)
      files+=("${2:?"resyntax: --file needs a path"}")
      shift 2
      ;;
    *)
      base_ref=$1
      shift
      ;;
  esac
done
if ! [[ $timeout_ms =~ ^[0-9]+$ ]]; then
  echo "resyntax: the analyzer timeout is a whole number of milliseconds, not '$timeout_ms'" >&2
  exit 2
fi
if [ ${#files[@]} -gt 0 ] && [ -n "$base_ref" ]; then
  echo "resyntax: give a base ref or --file paths, not both" >&2
  exit 2
fi

suite=(--refactoring-suite "(file \"$PWD/scripts/resyntax-rules.rkt\")" cuml-recommendations)

pr_base() {
  if [ -n "${RESYNTAX_BASE:-}" ]; then
    git merge-base HEAD "$RESYNTAX_BASE" || {
      echo "resyntax: HEAD and RESYNTAX_BASE=$RESYNTAX_BASE have no merge base" >&2
      return 1
    }
    return
  fi
  if ! git remote get-url origin >/dev/null 2>&1; then
    git merge-base HEAD master
    return
  fi
  git fetch -q origin || echo "resyntax: git fetch failed; using the local origin refs" >&2
  local branch
  branch=$(gh pr view --json baseRefName -q .baseRefName 2>/dev/null) || branch=master
  git merge-base HEAD "origin/${branch:-master}" 2>/dev/null || git merge-base HEAD origin/master
}

cache_key() {
  {
    echo "$1"
    echo "$timeout_ms"
    git diff "$1" -- '*.rkt' '*.scrbl'
    raco pkg show --scope user --full-checksum resyntax
    cat scripts/resyntax.sh scripts/resyntax-rules.rkt
  } | sha256sum | cut -d' ' -f1
}

unanalysed() {
  sed -nE 's/^resyntax: analyzer ([^ ]+) (timed out after [0-9]+ ms|failed with error) while analyzing (.*[.](rkt|scrbl))(: .*)?$/\3 \1/p' |
    sed "s|^$PWD/||" | sort -u |
    awk '$1 == file { names = names ", " $2; next }
         { if (file != "") print file ": " names; file = $1; names = $2 }
         END { if (file != "") print file ": " names }'
}

report_unanalysed() {
  local listed=$1 file names line
  [ -n "$listed" ] || return 0
  echo "resyntax: warning: files not fully analysed: $(wc -l <<<"$listed"); an analyzer timed out at $timeout_ms ms or failed, so the rules that need it did not run on them; raise --analyzer-timeout or RESYNTAX_ANALYZER_TIMEOUT to cover them"
  while IFS= read -r line; do echo "  $line"; done <<<"$listed"
  if [ -n "${GITHUB_ACTIONS:-}" ]; then
    while IFS=: read -r file names; do
      echo "::warning file=$file::not fully analysed by Resyntax at $timeout_ms ms (${names# }); the rules that need those analyzers did not run here"
    done <<<"$listed"
  fi
}

key=""
if [ ${#files[@]} -gt 0 ]; then
  targets=()
  for file in "${files[@]}"; do
    targets+=(--file "$file")
  done
  what="the given files"
else
  raco test scripts/resyntax-rules-test.rkt scripts/resyntax-timeout-test.rkt
  if [ -z "$base_ref" ]; then
    base_ref=$(pr_base)
    key=$(cache_key "$base_ref")
    echo "resyntax: against $base_ref, the merge base with ${RESYNTAX_BASE:-the base branch of the PR}"
  fi
  targets=(--local-git-repository . "$base_ref")
  what=$base_ref
fi

if [ -n "$key" ] && [ "${RESYNTAX_NO_CACHE:-0}" != 1 ] && [ -e "$cache_dir/$key" ]; then
  echo "resyntax: unchanged since a run with no suggestions (key $key)"
  report_unanalysed "$(cat "$cache_dir/$key")"
  exit 0
fi

echo "resyntax: analyzer timeout $timeout_ms ms"
rc=0
out=$(resyntax analyze "${suite[@]}" "${targets[@]}" --analyzer-timeout "$timeout_ms" 2>&1) || rc=$?
echo "$out"
if [ "$rc" -ne 0 ]; then
  echo "resyntax exited $rc without completing the analysis" >&2
  exit 1
fi
listed=$(unanalysed <<<"$out")
report_unanalysed "$listed"
if grep -qE "resyntax: .*\.rkt:[0-9]+:[0-9]+ \[" <<<"$out"; then
  echo "resyntax found refactoring suggestions; run: resyntax fix ${suite[0]} '${suite[1]}' ${suite[2]} ${targets[*]} --analyzer-timeout $timeout_ms" >&2
  exit 1
fi
if [ -n "$listed" ]; then
  echo "resyntax: no suggestions against $what in the analysed files; files not fully analysed: $(wc -l <<<"$listed"), listed above"
else
  echo "resyntax: no suggestions against $what; every file fully analysed"
fi
if [ -n "$key" ]; then
  mkdir -p "$cache_dir"
  printf '%s' "$listed" >"$cache_dir/$key"
fi
