#lang racket/base

(require (only-in cuml fit pca transform)
         (only-in cuml/decomposition
                  pca-fit
                  pca-inverse-transform
                  pca-params
                  pca-result
                  pca-result-components
                  pca-result-explained-variance
                  pca-result-explained-variance-ratio
                  pca-result-mean
                  pca-result-noise-variance
                  pca-result-singular-values
                  pca-transform)
         (only-in math/flonum +max.0)
         (only-in racket/list make-list)
         (only-in racket/match match-lambda)
         (only-in racket/math nan?)
         (only-in rackunit check-equal? check-false check-true test-case)
         (only-in raft
                  device-matrix->list*
                  device-vector->list
                  list*->device-matrix
                  list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/cuml-error.rkt" check-cuml-error raised)
         (only-in "private/pca-data.rkt" magnitude-rows scaled-rows)
         (only-in "private/scaler-data.rkt" rows->matrix))

(define axes '((4 0) (-4 0) (0 1) (0 -1) (0 0)))
(define exact-axes '((4.0 0.0) (-4.0 0.0) (0.0 1.0) (0.0 -1.0) (0.0 0.0)))

(define (X)
  (list*->device-matrix axes #:dtype 'float64))

(define (same? got want)
  (and (= (length got) (length want))
       (for/and ([g (in-list got)]
                 [w (in-list want)])
         (if (list? w)
             (same? g w)
             (= g w)))))

(test-gpu "the manual: the axis example's components, spectrum and transform"
  (define fitted (pca-fit (X)))
  (check-equal? (device-matrix->list* (pca-result-components fitted)) '((1.0 0.0) (0.0 1.0)))
  (check-equal? (map device-vector->list
                     (list (pca-result-explained-variance fitted)
                           (pca-result-explained-variance-ratio fitted)
                           (pca-result-singular-values fitted)))
                (list '(8.0 0.5) (list (/ 8.0 8.5) (/ 0.5 8.5)) (list (sqrt 32.0) (sqrt 2.0))))
  (check-true (same? (device-matrix->list* (pca-transform fitted (X))) exact-axes))
  (check-true
   (same? (~> (pca-transform fitted (X)) (pca-inverse-transform fitted _) device-matrix->list*)
          exact-axes)))

(test-gpu "the manual: noise variance and the fit's refusals"
  (check-equal? (pca-result-noise-variance (pca-fit (X) #:n-components 1)) 0.5)
  (check-cuml-error 'logic
                    "pca-fit: n-components: expected at most 2, got 3"
                    (lambda () (pca-fit (X) #:n-components 3)))
  (check-cuml-error 'logic
                    "pca-fit: X: expected at least 2 distinct rows, got 1"
                    (lambda () (pca-fit (list*->device-matrix '((1 2) (1 2)) #:dtype 'float64)))))

(test-gpu "the manual: whitening, and a transform refusal"
  (define whitened (~> (pca-fit (X) #:whiten #t) (pca-transform (X)) device-matrix->list*))
  (check-true (same? whitened
                     (for/list ([row (in-list exact-axes)])
                       (map / row (list (sqrt 8.0) (sqrt 0.5))))))
  (check-cuml-error
   'logic
   "pca-transform: X: expected 2 columns, got 3"
   (lambda () (pca-transform (pca-fit (X)) (list*->device-matrix '((1 2 3)) #:dtype 'float64)))))

(test-gpu "the manual: pca-params and the estimator"
  (check-equal? (hash-ref (pca-params #:solver 'jacobi) 'solver) 'jacobi)
  (check-cuml-error 'logic
                    "pca-params: pca has no parameter copy"
                    (lambda () (pca-params (hasheq 'copy #t))))
  (check-true (same? (~> (pca #:n-components 1) (fit (X)) (transform (X)) device-matrix->list*)
                     '((4.0) (-4.0) (0.0) (0.0) (0.0))))
  (check-cuml-error 'logic "transform: pca is not fitted" (lambda () (transform (pca) (X)))))

(define most +max.0)
(define margin-width (* 4 2))
(define slack 1e-12)

(define (bound-cases plain whitened)
  (define (huge v)
    (list*->device-matrix `((,v 0)) #:dtype 'float64))
  `(("fit" ,(sqrt (/ most (* margin-width 2)))
           ,(lambda (v) (pca-fit (list*->device-matrix `((,v 0) (0 0)) #:dtype 'float64))))
    ("tol" 3.4028234663852886e+38 ,(lambda (v) (pca-fit (X) #:tol v)))
    ("transform" ,(/ most margin-width) ,(lambda (v) (pca-transform plain (huge v))))
    ("whitened transform" ,(* (/ most margin-width) (min 1 (sqrt 0.5)))
                          ,(lambda (v) (pca-transform whitened (huge v))))
    ("inverse" ,(/ most margin-width) ,(lambda (v) (pca-inverse-transform plain (huge v))))
    ("whitened inverse" ,(/ (/ most margin-width) (max 1 (sqrt 8.0)))
                        ,(lambda (v) (pca-inverse-transform whitened (huge v))))))

(test-gpu "the manual: the documented bounds are the refused ones"
  (for-each (match-lambda
              [(list what documented call)
               (test-case what
                 (check-cuml-error 'logic
                                   #rx"expected a magnitude of at most|expected at most"
                                   (lambda () (call (* documented (+ 1 slack)))))
                 (check-false (raised (lambda () (call (* documented (- 1 slack)))))))])
            (bound-cases (pca-fit (X)) (pca-fit (X) #:whiten #t))))

(test-gpu "the manual: a hand-built result is checked only for dtypes and shapes"
  (define fitted (pca-fit (X) #:whiten #t))
  (define hand-built
    (pca-result (pca-result-components fitted)
                (list->device-vector '(-1.0 0.5) #:dtype 'float64)
                (pca-result-explained-variance-ratio fitted)
                (pca-result-singular-values fitted)
                (pca-result-mean fitted)
                0.0
                5
                #t))
  (check-true (~> (pca-transform hand-built (list*->device-matrix '((1 1)) #:dtype 'float64))
                  device-matrix->list*
                  caar
                  nan?)))

(test-gpu "the manual: the whitened round trip needs every root of variance at least epsilon"
  (define (round-trip rows)
    (define X (rows->matrix rows 'float32))
    (define fitted (pca-fit X #:whiten #t))
    (values (~> (pca-transform fitted X) (pca-inverse-transform fitted _) device-matrix->list*)
            (map sqrt (device-vector->list (pca-result-explained-variance fitted)))
            (device-vector->list (pca-result-mean fitted))))
  (define-values (unit unit-roots _unit-mean) (round-trip magnitude-rows))
  (check-true (andmap (lambda (root) (>= root 1.1920928955078125e-07)) unit-roots))
  (check-true (for*/and ([(row want) (in-parallel (in-list unit) (in-list magnitude-rows))]
                         [(x y) (in-parallel (in-list row) (in-list want))])
                (<= (abs (- x y)) 1e-5)))
  (define-values (tiny tiny-roots tiny-mean) (round-trip (scaled-rows magnitude-rows 1e-25)))
  (check-true (andmap (lambda (root) (< root 1.1920928955078125e-07)) tiny-roots))
  (check-equal? tiny (make-list (length magnitude-rows) tiny-mean)))
