#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mapfile -t files < <(find .fmt.rkt cuml-*/ lint scripts -name '*.rkt' -not -path '*/compiled/*' | sort)
raco review "${files[@]}"
echo "review: ${#files[@]} files clean"
mapfile -t libraries < <(find cuml-*-lib -type d -name private -not -path '*/compiled/*' | sort)
racket lint/duplicates.rkt cuml-test:cuml-glmnet-test "$(
  IFS=:
  echo "${libraries[*]}"
)" lint:scripts
echo "review: no definition copied between test files, between library private modules, or between lint and script modules"
racket lint/licence.rkt cuml-test cuml-glmnet-test cuml-glmnet-test glmnet
echo "review: cuml-test requires nothing from cuml-glmnet-test or glmnet"
raco test lint
