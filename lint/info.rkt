#lang info

#|review: ignore|#

(define collection "cuml-lint")
(define deps '("base" "review"))
(define build-deps '("rackunit-lib"))
(define review-exts '((cuml-lint/review should-review? review-syntax)))
(define pkg-desc "raco review rules for the cuml packages' test forms")
(define license 'Apache-2.0)
