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
         (only-in rackunit check-equal? check-false check-true test-case)
         (only-in raft device-matrix? device-vector? dtype layout shape)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/scaler-data.rkt" normal-rows rows->matrix))

(define (vector-shape v)
  (list (device-vector? v) (dtype v) (shape v)))

(define (matrix-shape m)
  (list (device-matrix? m) (dtype m) (layout m) (shape m)))

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
      (check-true (and (exact-integer? (standard-scaler-result-n-samples fitted))
                       (exact-integer? (standard-scaler-result-n-features fitted))))
      (define T (standard-scaler-transform fitted X))
      (define I (standard-scaler-inverse-transform fitted T))
      (define-values (again T2) (standard-scaler-fit-transform X))
      (for ([out (list T I T2)])
        (check-equal? (matrix-shape out) (list #t dt 'row-major (list n d)))
        (check-false (eq? out X)))
      (check-true (standard-scaler-result? again))
      (check-false (eq? T I)))))

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
