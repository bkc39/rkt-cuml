#lang racket/base

(require (only-in cuml/linear-model mbsgd-regressor-fit)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/common-data.rkt" on-vector)
         (only-in "private/linear-data.rkt" law-case on-matrix)
         (only-in "private/mbsgd-data.rkt" normal-rows reg-target)
         (only-in "private/mbsgd-gate-data.rkt" rose)
         (only-in "private/refusal-rows.rkt" check-row row row-id))

(define not-finite #rx"^the solution is not finite; lower #:eta0 \\(now 10\\)$")

(define rows (normal-rows 100 3 41))
(define targets (reg-target rows 41))

(define (diverging dt lr)
  (lambda ()
    (mbsgd-regressor-fit (on-matrix rows dt) (on-vector targets dt) #:eta0 10 #:learning-rate lr)))

(test-gpu "a diverging fit is refused: a float32 NaN as not finite, a huge float64 as a rise"
  (for ([r (list (row "float32 constant" not-finite (diverging 'float32 'constant))
                 (row "float64 constant" (rose "10") (diverging 'float64 'constant))
                 (row "float32 invscaling" not-finite (diverging 'float32 'invscaling))
                 (row "float64 invscaling" (rose "10") (diverging 'float64 'invscaling))
                 (row "float32 adaptive" not-finite (diverging 'float32 'adaptive)))])
    (law-case (row-id r)
      (check-row "mbsgd-regressor-fit" r))))
