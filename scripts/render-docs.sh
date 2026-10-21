#!/usr/bin/env bash
#
# Renders the manual to <dest>/cuml/ and fails on any warning Scribble prints
# (raco scribble exits 0 on an undefined tag or a duplicate definition), on
# any broken link left in the HTML, and on a for-label name that renders
# without a link (scripts/check-doc-links.rkt). Run inside `nix develop`.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

dest=${1:?usage: scripts/render-docs.sh <dest-dir>}
mkdir -p "$dest"
log=$(mktemp)
trap 'rm -f "$log"' EXIT

raco make -v cuml-doc/cuml/scribblings/cuml.scrbl
raco scribble --htmls ++xref-in setup/xref load-collections-xref \
  --dest "$dest" cuml-doc/cuml/scribblings/cuml.scrbl 2>&1 | tee "$log"

status=0
if grep -v "^Fontconfig warning: " "$log" | grep -iE "undefined tag|badlink|multiple times|warning"; then
  echo "render-docs: Scribble warned (above)" >&2
  status=1
fi
if grep -rl 'class="badlink"' "$dest/cuml"; then
  echo "render-docs: broken links in the pages above" >&2
  status=1
fi
if ! racket scripts/check-doc-links.rkt "$dest/cuml"; then
  echo "render-docs: missing or wrong links in the pages above" >&2
  status=1
fi
printf '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=cuml/index.html"><a href="cuml/index.html">cuML: GPU machine learning for Racket</a>\n' \
  >"$dest/index.html"
[ "$status" -eq 0 ] && echo "render-docs: $dest/cuml/index.html"
exit "$status"
