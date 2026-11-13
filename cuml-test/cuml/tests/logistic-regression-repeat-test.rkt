#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes)
         (only-in "private/logistic-data.rkt" blob-case case-fit fit-parts lr-case-name relative-gap))

(define small (list (blob-case "binary" 200 4 2 51) (blob-case "four classes" 200 4 4 52)))
(define large (list (blob-case "binary" 2000 6 2 53) (blob-case "four classes" 2000 6 4 54)))

(define (repeat-bound dt)
  (if (eq? dt 'float32) 1e-3 1e-12))

(define (flat parts)
  (match-define (list coef intercept _) parts)
  (append (apply append coef) intercept))

(test-gpu "a fit of 200 rows repeats bit for bit, and one of 2000 within 1e-3 (float32) and 1e-12 (float64) (S3)"
  (for* ([dt (in-list dtypes)]
         [c (in-list small)])
    (test-case (format "~a ~a small" dt (lr-case-name c))
      (check-equal? (fit-parts (case-fit c dt)) (fit-parts (case-fit c dt)))))
  (for* ([dt (in-list dtypes)]
         [c (in-list large)])
    (test-case (format "~a ~a large" dt (lr-case-name c))
      (define gap (relative-gap (flat (fit-parts (case-fit c dt))) (flat (fit-parts (case-fit c dt)))))
      (check-true (<= gap (repeat-bound dt)) (format "gap ~a" gap)))))
