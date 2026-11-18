#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mapfile -t files < <(find .fmt.rkt cuml-*/ lint scripts -name '*.rkt' -not -path '*/compiled/*' | sort)
raco review "${files[@]}"
echo "review: ${#files[@]} files clean"
mapfile -t libraries < <(find cuml-*-lib -type d -name private -not -path '*/compiled/*' | sort)
racket lint/duplicates.rkt cuml-test:cuml-glmnet-test "$(IFS=:; echo "${libraries[*]}")"
echo "review: no definition copied between test files, or between library private modules"
raco test lint
