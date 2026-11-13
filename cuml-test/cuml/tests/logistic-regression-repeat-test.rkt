#lang racket/base

(require (only-in rackunit check-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" by-dtype dtypes)
         (only-in "private/logistic-data.rkt"
                  blob-case
                  case-fit
                  fit-flat
                  fit-parts
                  lr-case-name
                  relative-gap))

(define small (list (blob-case "binary" 200 4 2 51) (blob-case "four classes" 200 4 4 52)))
(define large (list (blob-case "binary" 2000 6 2 53) (blob-case "four classes" 2000 6 4 54)))

(define (repeat-gap-bound dt)
  (by-dtype dt 1e-3 1e-12))

(test-gpu "a small fit repeats bit for bit, and a large one nearly"
  (for* ([dt (in-list dtypes)]
         [c (in-list small)])
    (test-case (format "~a ~a small" dt (lr-case-name c))
      (check-equal? (fit-parts (case-fit c dt)) (fit-parts (case-fit c dt)))))
  (for* ([dt (in-list dtypes)]
         [c (in-list large)])
    (test-case (format "~a ~a large" dt (lr-case-name c))
      (define gap
        (relative-gap (fit-flat (fit-parts (case-fit c dt))) (fit-flat (fit-parts (case-fit c dt)))))
      (check-true (<= gap (repeat-gap-bound dt)) (format "gap ~a" gap)))))
