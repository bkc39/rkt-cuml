#lang racket/base

(require (only-in cuml/linear-model logistic-regression-decision-function)
         (only-in racket/match match match-define)
         (only-in rackunit check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in "private/linear-data.rkt" dtypes on-matrix)
         (only-in "private/linear-twin-data.rkt" logistic-twin logistic-twin-results)
         (only-in "private/logistic-data.rkt"
                  argmax-index
                  blob-case
                  case-fit
                  centred
                  fit-parts
                  lr-case-name
                  lr-case-X
                  relative-gap)
         (only-in "private/python-env.rkt" test-twin))

(define cases
  (list (blob-case "binary" 100 3 2 31)
        (blob-case "binary weighted" 100 3 2 32 #:weighted? #t)
        (blob-case "three classes" 120 3 3 33)))

(define settings '((l2 1.0 #f) (l1 1.0 #f) (elasticnet 0.5 0.5) (none 1.0 #f)))

(define (tight-tol dt)
  (if (eq? dt 'float32) 1e-6 1e-10))

(define (coef-bound dt)
  (if (eq? dt 'float32) 2e-3 1e-6))

(define (s4-cap dt)
  (if (eq? dt 'float32) 8e-2 3e-2))

(define rows
  (for*/list ([dt (in-list dtypes)]
              [c (in-list cases)]
              [s (in-list settings)])
    (list dt c s)))

(define (request row)
  (match-define (list dt c (list penalty C ratio)) row)
  (logistic-twin "sklearn-fit"
                 c
                 dt
                 (hasheq 'penalty (symbol->string penalty) 'C C 'l1_ratio (or ratio 'null) 'tol (tight-tol dt))))

(define (gauged coef intercept penalty)
  (if (= 1 (length coef))
      (append (car coef) intercept)
      (append (apply append (if (memq penalty '(l1 none)) (centred coef) coef)) (centred intercept))))

(define (margin z)
  (match (sort z >)
    [(list top) (abs top)]
    [(list* top next _) (- top next)]))

(define (label-of z)
  (if (= 1 (length z))
      (if (> (car z) 0) 1 0)
      (argmax-index z)))

(test-twin "sklearn's optimum within 2e-3 (float32) and 1e-6 (float64), its predictions off the boundary, and within 8e-2 and 3e-2 at the default tol (T3, S4)"
  (for ([row (in-list rows)]
        [theirs (in-list (logistic-twin-results (map request rows)))])
    (match-define (list dt c (list penalty C ratio)) row)
    (test-case (format "~a ~a ~a" dt (lr-case-name c) penalty)
      (define r
        (case-fit c dt #:penalty penalty #:C C #:l1-ratio ratio #:tol (tight-tol dt) #:max-iter 100000))
      (match-define (list coef intercept _) (fit-parts r))
      (define gap
        (relative-gap (gauged coef intercept penalty)
                      (gauged (hash-ref theirs 'coef) (hash-ref theirs 'intercept) penalty)))
      (check-true (<= gap (coef-bound dt)) (format "coef gap ~a" gap))
      (match-define (list default-coef default-intercept _)
        (fit-parts (case-fit c dt #:penalty penalty #:C C #:l1-ratio ratio)))
      (define default-gap
        (relative-gap (gauged default-coef default-intercept penalty)
                      (gauged (hash-ref theirs 'coef) (hash-ref theirs 'intercept) penalty)))
      (check-true (<= default-gap (s4-cap dt)) (format "default-tol gap ~a" default-gap))
      (define z (device-matrix->list* (logistic-regression-decision-function r (on-matrix (lr-case-X c) dt))))
      (for ([zi (in-list z)]
            [label (in-list (hash-ref theirs 'predict))]
            #:unless (< (margin zi) 1e-6))
        (check-true (= (label-of zi) label) "predict")))))
