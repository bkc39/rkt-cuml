#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

work=$(mktemp -d)
marker="gpu-suite-signals-$$-$RANDOM"

survivors() {
  local e
  { grep -lsz "^CUML_SIGNAL_PROBE=$marker\$" /proc/[0-9]*/environ || true; } | while read -r e; do
    e=${e#/proc/}
    echo "${e%/environ}"
  done
}

reap() {
  local p
  for p in $(survivors); do
    kill -KILL "$p" 2>/dev/null || true
  done
}

trap 'reap; rm -rf "$work"' EXIT

stubs="$work/stubs"
mkdir -p "$stubs"
gtests=(cumlrkt_cluster_tests cumlrkt_preprocessing_tests)
for t in "${gtests[@]}"; do
  printf '#!%s\ntouch %q\nexec sleep 60\n' "$BASH" "$work/started" >"$stubs/$t"
  chmod +x "$stubs/$t"
done

failed=0
for sig in INT TERM HUP; do
  rm -f "$work/started"
  CUML_SIGNAL_PROBE=$marker CUML_SHIM_TESTS=$stubs CUML_SHIM_SANITIZED_TESTS=$stubs \
    env --default-signal=INT "$BASH" scripts/gpu-suite.sh >"$work/out-$sig" 2>&1 &
  outer=$!
  for _ in $(seq 50); do
    [ ! -e "$work/started" ] || break
    sleep 0.1
  done
  if [ ! -e "$work/started" ]; then
    echo "gpu-suite-signals: the stub suite never started" >&2
    cat "$work/out-$sig" >&2
    exit 1
  fi
  if [ -z "$(survivors | grep -vx "$outer" || true)" ]; then
    echo "gpu-suite-signals: no suite process carries the probe marker" >&2
    exit 1
  fi
  kill -"$sig" "$outer"
  for _ in $(seq 20); do
    [ -n "$(survivors)" ] || break
    sleep 0.1
  done 2>/dev/null
  left=$(survivors | tr '\n' ' ')
  reap
  rc=0
  wait "$outer" 2>/dev/null || rc=$?
  if [ -n "$left" ]; then
    echo "gpu-suite-signals: SIG$sig to gpu-suite.sh left pids $left running after 2 s" >&2
    failed=1
  fi
  if [ "$rc" -ne $((128 + $(kill -l "$sig"))) ]; then
    echo "gpu-suite-signals: SIG$sig to gpu-suite.sh exited $rc, not by SIG$sig" >&2
    cat "$work/out-$sig" >&2
    failed=1
  fi
done

lsan_tree="$work/tree"
mkdir -p "$lsan_tree/scripts" "$work/plain" "$work/sanitized"
cp scripts/gpu-suite.sh "$lsan_tree/scripts/"
for t in "${gtests[@]}"; do
  printf '#!%s\nexit 0\n' "$BASH" >"$work/plain/$t"
  # shellcheck disable=SC2016 # the stub, not this script, expands it
  printf '#!%s\necho "${LSAN_OPTIONS-unset}" >%q\nexit 3\n' "$BASH" "$work/lsan-seen" >"$work/sanitized/$t"
  chmod +x "$work/plain/$t" "$work/sanitized/$t"
done
lsan_seen() {
  rm -f "$work/lsan-seen"
  env -u LSAN_OPTIONS CUML_GPU_SUITE_BOUNDED=1 CUML_SHIM_TESTS="$work/plain" \
    CUML_SHIM_SANITIZED_TESTS="$work/sanitized" CUML_ASAN_OPTIONS= CUML_UBSAN_OPTIONS= \
    "$BASH" "$lsan_tree/scripts/gpu-suite.sh" >"$work/out-lsan" 2>&1 || true
  cat "$work/lsan-seen" 2>/dev/null || echo "never ran"
}
seen=$(lsan_seen)
if [ "$seen" != unset ]; then
  echo "gpu-suite-signals: without shim/sanitizers/lsan.supp the sanitized gtests saw LSAN_OPTIONS=$seen" >&2
  cat "$work/out-lsan" >&2
  failed=1
fi
mkdir -p "$lsan_tree/shim/sanitizers"
touch "$lsan_tree/shim/sanitizers/lsan.supp"
seen=$(lsan_seen)
if [ "$seen" != "suppressions=$lsan_tree/shim/sanitizers/lsan.supp" ]; then
  echo "gpu-suite-signals: with shim/sanitizers/lsan.supp the sanitized gtests saw LSAN_OPTIONS=$seen" >&2
  cat "$work/out-lsan" >&2
  failed=1
fi

[ "$failed" = 0 ] || exit 1
echo "gpu-suite-signals: SIGINT, SIGTERM and SIGHUP stop the whole suite and re-raise"
echo "gpu-suite-signals: the sanitized gtests get LSAN_OPTIONS=suppressions=shim/sanitizers/lsan.supp when it exists"
