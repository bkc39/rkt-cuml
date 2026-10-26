#lang racket/base

(require (only-in cuml/decomposition
                  pca-fit
                  pca-result-components
                  pca-result-explained-variance
                  pca-result-explained-variance-ratio
                  pca-result-singular-values
                  tsvd-fit)
         (only-in racket/list append*)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/pca-data.rkt" centred column fixture-name fixture-rows)
         (only-in "private/scaler-data.rkt" round-to rows->matrix unit-roundoff)
         (only-in "private/tsvd-data.rkt"
                  accuracy-problems
                  exact
                  exact-reference
                  first-problems
                  geomspace
                  householder-fixtures
                  normal-rows-of
                  svd
                  tsvd-of
                  worst-difference))

(test-gpu "L8: the fit is within c u F^2 of the Householder fixtures' exact SVD"
  (for* ([e (in-list householder-fixtures)]
         [dt (in-list '(float32 float64))]
         [solver (in-list '(full jacobi))]
         [k (in-list '(6 12))])
    (match-define (exact f spectrum components) e)
    (test-case (format "~a ~a ~a k=~a" (fixture-name f) dt solver k)
      (define rows (fixture-rows f dt))
      (define fitted (tsvd-of (tsvd-fit (rows->matrix rows dt) #:n-components k #:solver solver)))
      (define c (if (eq? solver 'jacobi) 256 64))
      (check-equal?
       (first-problems
        (accuracy-problems fitted (exact-reference rows spectrum components k) rows dt c))
       '()))))

(define (centred-rows seed n d dt)
  (define scales (geomspace 10.0 d))
  (define rows
    (for/list ([row (in-list (normal-rows-of seed n d))])
      (map (lambda (z s) (+ 5.0 (* z s))) row scales)))
  (define means
    (for/list ([j (in-range d)])
      (/ (apply + (column rows j)) n)))
  (define narrow (round-to dt))
  (for/list ([row (in-list rows)])
    (map narrow (centred row means))))

(test-gpu "L14: on centred X, the fit is pca-fit's, its variances scaled by (n - 1)/n"
  (for* ([shape (in-list '((500 12 5) (300 6 6)))]
         [dt (in-list '(float32 float64))])
    (match-define (list n d k) shape)
    (test-case (format "~ax~a k=~a ~a" n d k dt)
      (define X (rows->matrix (centred-rows (+ n d) n d dt) dt))
      (match-define (svd components variance ratio singular) (tsvd-of (tsvd-fit X #:n-components k)))
      (define pca (pca-fit X #:n-components k))
      (define pca-variance (device-vector->list (pca-result-explained-variance pca)))
      (define u (unit-roundoff dt))
      (define worst-component
        (worst-difference (append* components)
                          (append* (device-matrix->list* (pca-result-components pca)))))
      (check-true (<= worst-component (if (eq? dt 'float32) 1e-5 1e-12))
                  (format "components ~a" worst-component))
      (define pca-singular (device-vector->list (pca-result-singular-values pca)))
      (match-define (cons largest-singular _) pca-singular)
      (check-true (<= (worst-difference singular pca-singular) (* 64 u largest-singular))
                  (format "singular values ~a vs ~a" singular pca-singular))
      (define scaled
        (for/list ([v (in-list pca-variance)])
          (/ (* v (- n 1)) n)))
      (match-define (cons largest-variance _) scaled)
      (check-true (<= (worst-difference variance scaled) (* 128 u largest-variance))
                  (format "variances ~a vs ~a" variance scaled))
      (check-true
       (<= (worst-difference ratio (device-vector->list (pca-result-explained-variance-ratio pca)))
           (* 64 u))
       "ratios"))))
