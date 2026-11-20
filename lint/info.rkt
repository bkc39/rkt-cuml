#lang info

#|review: ignore|#

(define collection "cuml-lint")
(define deps '("base" "review"))
(define build-deps '("rackunit-lib"))
(define test-timeouts '(("tests/failure-scope-fuzz-test.rkt" 120)))
(define review-exts '((cuml-lint/review should-review? review-syntax)))
(define pkg-desc "raco review rules for the cuml packages' test forms")
(define license 'Apache-2.0)
