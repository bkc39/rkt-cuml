#lang racket/base

(require (only-in cuml/linear-model
                  logistic-regression-decision-function
                  logistic-regression-predict
                  logistic-regression-predict-proba)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-vector->list)
         (only-in "private/linear-data.rkt" on-matrix unit-roundoff)
         (only-in "private/linear-twin-data.rkt" decoded)
         (only-in "private/logistic-data.rkt" fit-flat fit-parts flat-matrix lr-case-X lr-fit-n-iter)
         (only-in "private/logistic-twin-data.rkt" logistic-twin-results their-fit twin-trial)
         (only-in "private/logistic-twin-rows.rkt"
                  classes-of
                  repeatable?
                  request
                  row-fit
                  row-name
                  rows)
         (only-in "private/python-env.rkt" test-twin))

(define (proba-gap ours theirs k)
  (for/fold ([worst 0.0])
            ([o (in-list ours)]
             [t (in-list theirs)])
    (max worst
         (/ (abs (- o t))
            (if (= k 2)
                1.0
                (* (+ k 3) t (+ 1 (abs (log t)))))))))

(test-twin "a repeatable fit is Python cuML's bit for bit"
  (define kept (filter repeatable? rows))
  (for ([r (in-list kept)]
        [theirs (in-list (logistic-twin-results (map request kept)))])
    (match-define (twin-trial dt c _ _) r)
    (test-case (row-name r)
      (define X (on-matrix (lr-case-X c) dt))
      (define fitted (row-fit r))
      (check-equal? (fit-flat (fit-parts fitted)) (fit-flat (their-fit theirs dt)) "fit")
      (check-equal? (lr-fit-n-iter (fit-parts fitted)) (hash-ref theirs 'n_iter) "n-iter")
      (check-equal? (flat-matrix (logistic-regression-decision-function fitted X))
                    (decoded dt (hash-ref theirs 'decision_bits))
                    "decision")
      (check-equal? (device-vector->list (logistic-regression-predict fitted X))
                    (hash-ref theirs 'predict)
                    "predict")
      (define k (classes-of c))
      (define proba
        (decoded (string->symbol (hash-ref theirs 'proba_dtype)) (hash-ref theirs 'proba_bits)))
      (define bound
        (if (and (= k 2) (eq? dt 'float32))
            (expt 2.0 -23)
            (unit-roundoff dt)))
      (define gap (proba-gap (flat-matrix (logistic-regression-predict-proba fitted X)) proba k))
      (check-true (<= gap bound) (format "proba gap ~a, ~a of the bound" gap (/ gap bound))))))
