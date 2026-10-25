#lang racket/base

(require (only-in cuml/preprocessing
                  standard-scaler-fit
                  standard-scaler-fit-transform
                  standard-scaler-inverse-transform
                  standard-scaler-result-mean
                  standard-scaler-result-n-features
                  standard-scaler-result-n-samples
                  standard-scaler-result-scale
                  standard-scaler-result-var
                  standard-scaler-result-with-mean
                  standard-scaler-result-with-std
                  standard-scaler-result?
                  standard-scaler-transform)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list* device-vector->list list*->device-matrix)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/scaler-data.rkt" matrix-shape normal-rows rows->matrix vector-shape))

(test-gpu "L1: the result's vectors are float64 of length d, and the outputs keep X's shape and dtype"
  (for* ([dt (in-list '(float32 float64))]
         [size (in-list '((1 1) (7 13) (256 64)))])
    (match-define (list n d) size)
    (test-case (format "~a ~ax~a" dt n d)
      (define X (rows->matrix (normal-rows (+ n d) n d dt) dt))
      (define fitted (standard-scaler-fit X))
      (check-true (standard-scaler-result? fitted))
      (for ([v (list (standard-scaler-result-mean fitted)
                     (standard-scaler-result-var fitted)
                     (standard-scaler-result-scale fitted))])
        (check-equal? (vector-shape v) (list #t 'float64 (list d))))
      (check-equal? (list (standard-scaler-result-n-samples fitted)
                          (standard-scaler-result-n-features fitted)
                          (standard-scaler-result-with-mean fitted)
                          (standard-scaler-result-with-std fitted))
                    (list n d #t #t))
      (define T (standard-scaler-transform fitted X))
      (define I (standard-scaler-inverse-transform fitted T))
      (define-values (again T2) (standard-scaler-fit-transform X))
      (for ([out (list T I T2)])
        (check-equal? (matrix-shape out) (list #t dt 'row-major (list n d))))
      (check-true (standard-scaler-result? again)))))

(test-gpu "L1: the flags give the documented fields"
  (define X (rows->matrix (normal-rows 3 7 3 'float32) 'float32))
  (for* ([with-mean (in-list '(#t #f))]
         [with-std (in-list '(#t #f))])
    (define fitted (standard-scaler-fit X #:with-mean with-mean #:with-std with-std))
    (check-equal? (list (and (standard-scaler-result-mean fitted) #t)
                        (and (standard-scaler-result-var fitted) #t)
                        (and (standard-scaler-result-scale fitted) #t)
                        (standard-scaler-result-with-mean fitted)
                        (standard-scaler-result-with-std fitted)
                        (standard-scaler-result-n-features fitted))
                  (list (or with-mean with-std) with-std with-std with-mean with-std 3)
                  (format "~a ~a" with-mean with-std))))

(test-gpu "a scale that rounds to 0 in X's dtype gives non-finite output, which the inverse refuses"
  (define X (list*->device-matrix (list (list 0.0) (list (expt 2.0 -149))) #:dtype 'float32))
  (define fitted (standard-scaler-fit X))
  (check-equal? (device-vector->list (standard-scaler-result-scale fitted)) (list (expt 2.0 -150)))
  (define T (standard-scaler-transform fitted X))
  (check-equal? (device-matrix->list* T) '((+nan.0) (+inf.0)))
  (check-cuml-error
   'logic
   "standard-scaler-inverse-transform: X: row 0, column 0: expected a finite value, got nan"
   (lambda () (standard-scaler-inverse-transform fitted T))))
