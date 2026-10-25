#lang racket/base

(require (only-in cuml/decomposition
                  tsvd-fit
                  tsvd-fit-transform
                  tsvd-inverse-transform
                  tsvd-result-components
                  tsvd-result-explained-variance
                  tsvd-result-explained-variance-ratio
                  tsvd-result-singular-values
                  tsvd-result?
                  tsvd-transform)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-false check-true test-case)
         (only-in raft device-matrix->list* device-matrix? device-vector? dtype layout shape)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/pca-data.rkt" fixture-name fixture-rows)
         (only-in "private/scaler-data.rkt" bits-equal? rows->matrix)
         (only-in "private/tsvd-data.rkt" law-fixtures))

(define (vector-shape v)
  (list (device-vector? v) (dtype v) (shape v)))

(define (matrix-shape m)
  (list (device-matrix? m) (dtype m) (layout m) (shape m)))

(test-gpu "L1: the result's fields have their shapes and X's dtype, and the outputs keep it"
  (for* ([dt (in-list '(float32 float64))]
         [f (in-list law-fixtures)]
         [k (in-list '(1 2))])
    (define rows (fixture-rows f dt))
    (match-define (cons first-row _) rows)
    (define n (length rows))
    (define d (length first-row))
    (test-case (format "~a ~a k=~a" dt (fixture-name f) k)
      (define X (rows->matrix rows dt))
      (define result (tsvd-fit X #:n-components k))
      (check-true (tsvd-result? result))
      (check-equal? (matrix-shape (tsvd-result-components result)) (list #t dt 'row-major (list k d)))
      (for ([v (list (tsvd-result-explained-variance result)
                     (tsvd-result-explained-variance-ratio result)
                     (tsvd-result-singular-values result))])
        (check-equal? (vector-shape v) (list #t dt (list k))))
      (define T (tsvd-transform result X))
      (define I (tsvd-inverse-transform result T))
      (define-values (again T2) (tsvd-fit-transform X #:n-components k))
      (check-true (tsvd-result? again))
      (for ([out (list T T2)])
        (check-equal? (matrix-shape out) (list #t dt 'row-major (list n k))))
      (check-equal? (matrix-shape I) (list #t dt 'row-major (list n d)))
      (for ([out (list T I T2)])
        (check-false (eq? out X))))))

(test-gpu "L1: fit, transform and inverse leave the caller's arrays bitwise unchanged"
  (for* ([dt (in-list '(float32 float64))]
         [f (in-list law-fixtures)]
         [solver (in-list '(full jacobi))])
    (test-case (format "~a ~a ~a" dt (fixture-name f) solver)
      (define rows (fixture-rows f dt))
      (define X (rows->matrix rows dt))
      (define result (tsvd-fit X #:n-components 2 #:solver solver))
      (define components (device-matrix->list* (tsvd-result-components result)))
      (define T (tsvd-transform result X))
      (check-true (bits-equal? (device-matrix->list* X) rows) "fit and transform")
      (define before (device-matrix->list* T))
      (tsvd-inverse-transform result T)
      (check-true (bits-equal? (device-matrix->list* T) before) "inverse")
      (check-true (bits-equal? (device-matrix->list* (tsvd-result-components result)) components)
                  "components"))))
