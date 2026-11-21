#lang racket/base

(require (only-in cuml/linear-model
                  logistic-regression-decision-function
                  logistic-regression-predict
                  logistic-regression-predict-proba)
         (only-in racket/match match-define struct*)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-vector->list)
         (only-in threading ~>)
         (only-in "private/common-data.rkt" unit-roundoff)
         (only-in "private/linear-data.rkt" on-matrix)
         (only-in "private/logistic-data.rkt" fit-flat fit-parts flat-matrix lr-case-X lr-fit-n-iter)
         (only-in "private/logistic-twin-data.rkt" their-fit twin-trial)
         (only-in "private/logistic-twin-rows.rkt"
                  classes-of
                  float32-binary-bound
                  for-each-twin
                  repeatable?
                  request
                  row-fit
                  row-name
                  rows)
         (only-in "private/python-env.rkt" bits->floats test-twin))

(define (proba-scale t k)
  (if (= k 2)
      1.0
      (* (+ k 3) t (+ 1 (abs (log t))))))

(define (proba-gap ours theirs k)
  (for/fold ([worst 0.0])
            ([o (in-list ours)]
             [t (in-list theirs)])
    (max worst (/ (abs (- o t)) (proba-scale t k)))))

(define (proba-bound dt k)
  (if (and (= k 2) (eq? dt 'float32))
      float32-binary-bound
      (unit-roundoff dt)))

(test-twin "a repeatable fit is Python cuML's bit for bit"
  (for-each-twin
   (lambda (r theirs)
     (match-define (struct* twin-trial ([dt dt] (case c))) r)
     (define X (on-matrix (lr-case-X c) dt))
     (define k (classes-of c))
     (define proba
       (bits->floats (string->symbol (hash-ref theirs 'proba_dtype)) (hash-ref theirs 'proba_bits)))
     (define bound (proba-bound dt k))
     (test-case (row-name r)
       (define fitted (row-fit r))
       (check-equal? (fit-flat (fit-parts fitted)) (fit-flat (their-fit theirs dt)) "fit")
       (check-equal? (lr-fit-n-iter (fit-parts fitted)) (hash-ref theirs 'n_iter) "n-iter")
       (check-equal? (flat-matrix (logistic-regression-decision-function fitted X))
                     (bits->floats dt (hash-ref theirs 'decision_bits))
                     "decision")
       (check-equal? (device-vector->list (logistic-regression-predict fitted X))
                     (hash-ref theirs 'predict)
                     "predict")
       (define gap (~> (logistic-regression-predict-proba fitted X) flat-matrix (proba-gap proba k)))
       (check-true (<= gap bound) (format "proba gap ~a, ~a of the bound" gap (/ gap bound)))))
   (filter repeatable? rows)))
