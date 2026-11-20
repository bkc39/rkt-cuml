#lang racket/base

(require (only-in cuml/linear-model logistic-regression-decision-function)
         (only-in racket/match match match* match-define struct*)
         (only-in rackunit check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in "common-data.rkt" kept-rows)
         (only-in "linear-data.rkt" by-dtype dtypes on-matrix)
         (only-in "logistic-data.rkt"
                  decision-label
                  default-tol-cap
                  fit-parts
                  gauged-fit
                  gauged-sklearn
                  lr-case-name
                  lr-case-w
                  lr-case-X
                  relative-gap
                  repeated-rows
                  setting-fit
                  setting-penalty
                  tight-max-iter
                  tight-tol)
         (only-in "logistic-twin-data.rkt" logistic-twin-results setting-twin twin-trial))

(provide check-sklearn-twins
         collapsed
         sklearn-trials)

(define (optimum-gap-bound dt)
  (by-dtype dt 2e-3 1e-6))

(define boundary-margin 1e-6)

(define (sklearn-trials cases settings #:dtypes [dts dtypes])
  (for*/list ([dt (in-list dts)]
              [c (in-list cases)]
              [s (in-list settings)])
    (twin-trial dt c s #t)))

(define (oracle-case c repeated?)
  (if repeated?
      (repeated-rows c (lr-case-w c))
      c))

(define (collapsed labels counts)
  (match* (labels counts)
    [(_ '()) '()]
    [(_ (cons 0 rest)) (collapsed labels rest)]
    [((cons label _) (cons n rest)) (cons label (collapsed (list-tail labels n) rest))]))

(define (centre-coef? penalty)
  (and (memq penalty '(l1 none)) #t))

(define (margin z)
  (match (sort z >)
    [(list top) (abs top)]
    [(list* top next _) (- top next)]))

(define (check-sklearn-twins rows #:repeated? [repeated? #f])
  (define (request r)
    (match-define (struct* twin-trial ([dt dt] (case c) [setting s])) r)
    (setting-twin "sklearn-fit" (oracle-case c repeated?) dt s #:tol (tight-tol dt)))
  (for ([r (in-list rows)]
        [theirs (in-list (logistic-twin-results (map request rows)))])
    (match-define (struct* twin-trial ([dt dt] (case c) [setting s])) r)
    (define centred? (centre-coef? (setting-penalty s)))
    (define (gap-to-sklearn parts)
      (relative-gap (gauged-fit parts #:centre-coef? centred?)
                    (gauged-sklearn theirs #:centre-coef? centred?)))
    (define (tight-fit fitted-case)
      (setting-fit fitted-case dt s #:tol (tight-tol dt) #:max-iter tight-max-iter))
    (test-case (format "~a ~a ~a" dt (lr-case-name c) (setting-penalty s))
      (define fitted (tight-fit c))
      (define gap (gap-to-sklearn (fit-parts fitted)))
      (define solver-gap
        (if repeated?
            (gap-to-sklearn (fit-parts (tight-fit (oracle-case c #t))))
            0))
      (check-true (<= gap (+ solver-gap (optimum-gap-bound dt)))
                  (format "coef gap ~a, the unweighted fit's ~a" gap solver-gap))
      (define default-gap (gap-to-sklearn (fit-parts (setting-fit c dt s))))
      (check-true (<= default-gap (default-tol-cap dt)) (format "default-tol gap ~a" default-gap))
      (define all-z
        (device-matrix->list* (logistic-regression-decision-function fitted
                                                                     (on-matrix (lr-case-X c) dt))))
      (define-values (z labels)
        (if repeated?
            (values (kept-rows all-z (lr-case-w c))
                    (collapsed (hash-ref theirs 'predict) (lr-case-w c)))
            (values all-z (hash-ref theirs 'predict))))
      (for ([zi (in-list z)]
            [label (in-list labels)]
            #:unless (< (margin zi) boundary-margin))
        (check-true (= (decision-label zi) label) "predict")))))
