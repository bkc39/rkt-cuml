#lang racket/base

(require (only-in cuml/linear-model logistic-regression-decision-function)
         (only-in racket/match match match-define struct*)
         (only-in rackunit check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in "private/linear-data.rkt" by-dtype dtypes on-matrix)
         (only-in "private/logistic-data.rkt"
                  blob-case
                  decision-label
                  default-tol-cap
                  fit-parts
                  gauged
                  lr-case-name
                  lr-case-X
                  lr-fit
                  penalty-settings
                  relative-gap
                  setting-fit
                  setting-penalty
                  tight-max-iter
                  tight-tol)
         (only-in "private/logistic-twin-data.rkt" logistic-twin-results setting-twin)
         (only-in "private/python-env.rkt" test-twin))

(struct row (dt case setting))

(define cases
  (list (blob-case "binary" 100 3 2 31)
        (blob-case "binary weighted" 100 3 2 32 #:weighted? #t)
        (blob-case "three classes" 120 3 3 33)))

(define (optimum-gap-bound dt)
  (by-dtype dt 2e-3 1e-6))

(define boundary-margin 1e-6)

(define rows
  (for*/list ([dt (in-list dtypes)]
              [c (in-list cases)]
              [s (in-list penalty-settings)])
    (row dt c s)))

(define (request r)
  (match-define (row dt c s) r)
  (setting-twin "sklearn-fit" c dt s #:tol (tight-tol dt)))

(define (gauged-fit f penalty)
  (match-define (struct* lr-fit ([coef coef] [intercept intercept])) f)
  (gauged coef intercept #:centre-coef? (memq penalty '(l1 none))))

(define (gauged-sklearn theirs penalty)
  (gauged (hash-ref theirs 'coef)
          (hash-ref theirs 'intercept)
          #:centre-coef? (memq penalty '(l1 none))))

(define (margin z)
  (match (sort z >)
    [(list top) (abs top)]
    [(list* top next _) (- top next)]))

(test-twin "the fit is near sklearn's optimum and predicts its labels off the decision boundary"
  (for ([r (in-list rows)]
        [theirs (in-list (logistic-twin-results (map request rows)))])
    (match-define (row dt c s) r)
    (define penalty (setting-penalty s))
    (test-case (format "~a ~a ~a" dt (lr-case-name c) penalty)
      (define fitted (setting-fit c dt s #:tol (tight-tol dt) #:max-iter tight-max-iter))
      (define gap
        (relative-gap (gauged-fit (fit-parts fitted) penalty) (gauged-sklearn theirs penalty)))
      (check-true (<= gap (optimum-gap-bound dt)) (format "coef gap ~a" gap))
      (define default-gap
        (relative-gap (gauged-fit (fit-parts (setting-fit c dt s)) penalty)
                      (gauged-sklearn theirs penalty)))
      (check-true (<= default-gap (default-tol-cap dt)) (format "default-tol gap ~a" default-gap))
      (define z
        (device-matrix->list* (logistic-regression-decision-function fitted
                                                                     (on-matrix (lr-case-X c) dt))))
      (for ([zi (in-list z)]
            [label (in-list (hash-ref theirs 'predict))]
            #:unless (< (margin zi) boundary-margin))
        (check-true (= (decision-label zi) label) "predict")))))
