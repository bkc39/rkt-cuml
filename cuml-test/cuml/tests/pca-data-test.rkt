#lang racket/base

(require (only-in rackunit check-false check-true test-case)
         (only-in "private/pca-data.rkt" over-bound))

(test-case "over-bound reports a NaN difference as over the bound"
  (check-true (string? (over-bound "component 0" +nan.0 0.0 1e-3)))
  (check-true (string? (over-bound "component 0" 0.0 +nan.0 1e-3)))
  (check-false (over-bound "component 0" 1e-4 0.0 1e-3)))
