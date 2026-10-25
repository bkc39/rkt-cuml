#lang racket/base

(require (only-in cuml/decomposition
                  pca-fit
                  pca-fit-transform
                  pca-inverse-transform
                  pca-result-components
                  pca-result-explained-variance
                  pca-result-explained-variance-ratio
                  pca-result-mean
                  pca-result-n-samples
                  pca-result-noise-variance
                  pca-result-singular-values
                  pca-result-whiten
                  pca-result?
                  pca-transform)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-false check-pred check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/pca-data.rkt" fixture-name fixture-rows fixtures)
         (only-in "private/scaler-data.rkt" bits-equal? matrix-shape rows->matrix vector-shape))

(test-gpu "L1: the result's fields have their shapes and X's dtype, and the outputs keep it"
  (for* ([dt (in-list '(float32 float64))]
         [f (in-list fixtures)]
         [k (in-list '(#f 1 3))]
         [whiten (in-list '(#f #t))])
    (define rows (fixture-rows f dt 200))
    (match-define (cons first-row _) rows)
    (define n (length rows))
    (define d (length first-row))
    (define m (min n d))
    (define components (or k m))
    (test-case (format "~a ~a k=~a whiten=~a" dt (fixture-name f) k whiten)
      (define X (rows->matrix rows dt))
      (define result (pca-fit X #:n-components k #:whiten whiten))
      (check-true (pca-result? result))
      (check-equal? (matrix-shape (pca-result-components result))
                    (list #t dt 'row-major (list components d)))
      (for ([v (list (pca-result-explained-variance result)
                     (pca-result-explained-variance-ratio result)
                     (pca-result-singular-values result))])
        (check-equal? (vector-shape v) (list #t dt (list components))))
      (check-equal? (vector-shape (pca-result-mean result)) (list #t dt (list d)))
      (check-pred flonum? (pca-result-noise-variance result))
      (check-equal? (pca-result-n-samples result) n)
      (check-equal? (pca-result-whiten result) whiten)
      (define T (pca-transform result X))
      (define I (pca-inverse-transform result T))
      (define-values (again T2) (pca-fit-transform X #:n-components k #:whiten whiten))
      (check-true (pca-result? again))
      (for ([out (list T T2)])
        (check-equal? (matrix-shape out) (list #t dt 'row-major (list n components))))
      (check-equal? (matrix-shape I) (list #t dt 'row-major (list n d)))
      (for ([out (list T I T2)])
        (check-false (eq? out X))))))

(test-gpu "L1: fit, transform and inverse leave the caller's X bitwise unchanged"
  (for* ([dt (in-list '(float32 float64))]
         [f (in-list fixtures)]
         [solver (in-list '(full jacobi))])
    (test-case (format "~a ~a ~a" dt (fixture-name f) solver)
      (define rows (fixture-rows f dt 300))
      (define X (rows->matrix rows dt))
      (define result (pca-fit X #:solver solver #:whiten #t))
      (define T (pca-transform result X))
      (check-true (bits-equal? (device-matrix->list* X) rows) "fit")
      (define before (device-matrix->list* T))
      (pca-inverse-transform result T)
      (check-true (bits-equal? (device-matrix->list* X) rows) "transform")
      (check-true (bits-equal? (device-matrix->list* T) before) "inverse"))))
