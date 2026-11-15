#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-true)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/elastic-net-oracle.rkt" en-solution-coef oracle-elastic-net problem-of)
         (only-in "private/linear-data.rkt"
                  by-dtype
                  dtypes
                  law-case
                  oracle-lstsq
                  oracle-ridge
                  ridge-solution-coef
                  solution-coef)
         (only-in "private/logistic-newton.rkt" newton-coef)
         (only-in "private/mbsgd-data.rkt" blob-rows coef-gap normal-rows reg-target)
         (only-in "private/mbsgd-twin-data.rkt" clf-fit reg-fit))

(define rows (normal-rows 256 4 2))
(define targets (reg-target rows 2 #:noise 0.5))
(define n (length rows))
(define exact-rows (for/list ([r (in-list rows)]) (map inexact->exact r)))
(define exact-targets (map inexact->exact targets))
(define-values (blobs labels) (blob-rows 256 3 4 #:sep 1.0))

(define full-batch
  (hash '#:batch-size n '#:shuffle #f '#:tol 0.0 '#:eta0 0.05 '#:epochs 1000))

(define (ridge-optimum alpha)
  (ridge-solution-coef (oracle-ridge exact-rows exact-targets #f (* n alpha) #t)))

(define (en-optimum alpha ratio)
  (en-solution-coef (oracle-elastic-net (problem-of exact-rows exact-targets #f #t) alpha ratio)))

(define (check-gap name got optimum bound)
  (define gap (coef-gap got optimum))
  (check-true (<= gap bound) (format "~a: gap ~a, bound ~a" name gap bound))
  gap)

(define (hash-union base more)
  (for/fold ([all base]) ([(k v) (in-hash more)])
    (hash-set all k v)))

(define (reg-coef dt settings)
  (car (reg-fit dt rows targets (hash-union full-batch settings))))

(test-gpu "full-batch l2 descent converges to the ridge optimum, and its error falls with epochs"
  (define optimum (ridge-optimum 0.1))
  (for ([dt (in-list dtypes)])
    (law-case (format "~a" dt)
      (check-gap "1000 epochs"
                 (reg-coef dt (hash '#:penalty 'l2 '#:alpha 0.1))
                 optimum
                 (by-dtype dt 2e-6 1e-13))))
  (law-case "float64 errors at 10, 100 and 1000 epochs strictly decrease"
    (define gaps
      (for/list ([epochs (in-list '(10 100 1000))])
        (coef-gap (reg-coef 'float64 (hash '#:penalty 'l2 '#:alpha 0.1 '#:epochs epochs)) optimum)))
    (check-true (apply > gaps) (format "gaps ~a" gaps))))

(test-gpu "full-batch l1 descent converges to the lasso optimum at alpha/2"
  (define optimum (en-optimum 0.025 1))
  (for ([dt (in-list dtypes)])
    (law-case (format "~a" dt)
      (check-gap "l1" (reg-coef dt (hash '#:penalty 'l1 '#:alpha 0.05)) optimum (by-dtype dt 2e-6 1e-13)))))

(test-gpu "full-batch elastic-net descent converges to sklearn's ElasticNet at the mapped (A, L)"
  (define a 0.1)
  (define r 0.15)
  (define big (+ (* a r 1/2) (* a (- 1 r))))
  (define optimum (en-optimum big (/ (* a r 1/2) big)))
  (for ([dt (in-list dtypes)])
    (law-case (format "~a" dt)
      (check-gap "elasticnet"
                 (reg-coef dt (hash '#:penalty 'elasticnet '#:alpha a '#:l1-ratio r))
                 optimum
                 (by-dtype dt 2e-6 1e-13)))))

(test-gpu "full-batch unpenalised descent converges to least squares"
  (define optimum (solution-coef (oracle-lstsq exact-rows exact-targets #f #t)))
  (for ([dt (in-list dtypes)])
    (law-case (format "~a" dt)
      (check-gap "none" (reg-coef dt (hash '#:penalty 'none)) optimum (by-dtype dt 3e-6 1e-13)))))

(test-gpu "full-batch log descent without an intercept converges to the l2 logistic optimum"
  (define alpha 1e-2)
  (define optimum (newton-coef blobs labels #:lam (* 2 alpha) #:fit-intercept #f))
  (define settings
    (hash-union full-batch (hash '#:alpha alpha '#:eta0 0.5 '#:fit-intercept #f)))
  (for ([dt (in-list dtypes)])
    (law-case (format "~a" dt)
      (check-gap "log"
                 (car (clf-fit dt blobs labels 'log settings))
                 optimum
                 (by-dtype dt 5e-6 5e-8)))))

(test-gpu "minibatch descent of 32 rows sits within its noise floor of the ridge optimum"
  (define optimum (ridge-optimum 0.1))
  (define minibatch (hash '#:penalty 'l2 '#:alpha 0.1 '#:batch-size 32 '#:tol 0.0 '#:epochs 1000))
  (for* ([dt (in-list dtypes)]
         [arm (in-list (list (list "constant eta0 1e-3" (hash '#:eta0 1e-3) 5e-5)
                             (list "invscaling eta0 .05 power-t .5"
                                   (hash '#:learning-rate 'invscaling '#:eta0 0.05 '#:power-t 0.5)
                                   2e-5)
                             (list "constant eta0 .01" (hash '#:eta0 0.01) 2e-3)))])
    (match-define (list name settings bound) arm)
    (law-case (format "~a ~a" dt name)
      (check-gap name (car (reg-fit dt rows targets (hash-union minibatch settings))) optimum bound))))
