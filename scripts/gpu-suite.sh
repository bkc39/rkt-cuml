#!/usr/bin/env bash
set -euo pipefail

suite_seconds=1800
if [ -z "${CUML_GPU_SUITE_BOUNDED:-}" ]; then
  suite=
  sig=
  trap 'sig=INT; kill -TERM "${suite:-}" 2>/dev/null || true' INT
  trap 'sig=TERM; kill -TERM "${suite:-}" 2>/dev/null || true' TERM
  trap 'sig=HUP; kill -TERM "${suite:-}" 2>/dev/null || true' HUP
  CUML_GPU_SUITE_BOUNDED=1 timeout --kill-after=30 "$suite_seconds" "$BASH" "$0" "$@" &
  suite=$!
  [ -z "$sig" ] || kill -TERM "$suite" 2>/dev/null || true
  while :; do
    rc=0
    wait "$suite" || rc=$?
    kill -0 "$suite" 2>/dev/null || break
  done
  [ -z "$sig" ] || {
    trap - "$sig"
    kill -"$sig" $$
  }
  [ "$rc" -ne 124 ] || echo "gpu-suite: timed out after $suite_seconds s" >&2
  exit "$rc"
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${CUML_SHIM_TESTS:?run inside nix develop}"
: "${CUML_SHIM_SANITIZED_TESTS:?run inside nix develop}"

log=$(mktemp)
racket_log=$(mktemp)
trap 'rm -f "$log" "$racket_log"' EXIT

mapfile -t gtests < <(find "$CUML_SHIM_TESTS" -name 'cumlrkt_*_tests' | sort)
[ "${#gtests[@]}" -gt 0 ] || {
  echo "gpu-suite: no gtest binaries in $CUML_SHIM_TESTS" >&2
  exit 1
}

echo "== shim gtests"
for t in "${gtests[@]}"; do
  "$t" --gtest_brief=1 | tee -a "$log"
done

echo "== shim gtests under ASAN and UBSAN"
for t in "${gtests[@]}"; do
  ASAN_OPTIONS=$CUML_ASAN_OPTIONS UBSAN_OPTIONS=$CUML_UBSAN_OPTIONS \
    "$CUML_SHIM_SANITIZED_TESTS/$(basename "$t")" --gtest_brief=1 | tee -a "$log"
done

echo "== shim gtests under compute-sanitizer memcheck"
for t in "${gtests[@]}"; do
  compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 \
    --report-api-errors explicit "$t" --gtest_brief=1 | tee -a "$log"
done

echo "== LD_BIND_NOW load of each staged shim"
case ":$LD_LIBRARY_PATH:" in
  *:/usr/local/cuda*)
    echo "host CUDA is still on LD_LIBRARY_PATH" >&2
    exit 1
    ;;
esac
for lib in cuml-*-lib/cuml/*/native-libs/libcumlrkt_*.so; do
  LD_BIND_NOW=1 racket -l racket/base -l ffi/unsafe \
    -e "(void (ffi-lib (simplify-path (build-path \"$lib\"))))" \
    -e "(displayln \"$(basename "$lib") and libcuml: every symbol bound\")"
done

echo "== Racket tests"
raco make -v cuml-lib/cuml/main.rkt scripts/check-layout.rkt
raco test --process --timeout 45 cuml-test >"$racket_log" 2>&1 || {
  cat "$racket_log"
  exit 1
}
grep -E "^SKIP|^twin |^memory: |^device memory: |tests? passed|failure" "$racket_log"
cat "$racket_log" >>"$log"

echo "== binding census"
racket scripts/check-bindings.rkt | tee -a "$log"

mapfile -t allowed <scripts/allowed-skips.txt
unexpected=$(grep "^SKIP" "$log" | grep -v -E -f scripts/allowed-skips.txt || true)
if [ -n "$unexpected" ]; then
  echo "== cases that should have run here were skipped:" >&2
  echo "$unexpected" >&2
  exit 1
fi
no_gpu=$(grep -c -E -e "${allowed[0]}" "$log" || true)
mig=$(grep -c -E -e "${allowed[1]}" "$log" || true)
echo "== SKIP lines: $(grep -c "^SKIP" "$log" || true) ($no_gpu need no GPU, $mig no per-process memory: a MIG instance)"
