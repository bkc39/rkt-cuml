#lang racket/base

(require (only-in cuml/decomposition
                  pca-fit
                  pca-result-n-samples
                  pca-result-noise-variance
                  pca-transform)
         (only-in cuml/decomposition/private/native libcuml-version)
         (only-in racket/list append* take)
         (only-in racket/match match match-define)
         racket/runtime-path
         (only-in racket/sequence in-slice)
         (only-in rackunit check-equal?)
         (only-in raft device-matrix->list*)
         (only-in "pca-data.rkt"
                  fitted-components
                  fitted-mean
                  fitted-of
                  fitted-ratio
                  fitted-variance
                  fixture-name
                  fixture-rows
                  fixtures
                  norm)
         (only-in "python-env.rkt" run-twin)
         (only-in "scaler-data.rkt" rows->matrix))

(provide (struct-out ours)
         fit-ours
         l14-problems
         sample-count
         twin-case
         twin-key
         twin-ks
         twin-results
         twin-rows
         unit)

(define-runtime-path twin "../python/pca_twin.py")

(define most-elements 40000)

(define (twin-rows f dt)
  (define all (fixture-rows f dt))
  (match-define (cons first-row _) all)
  (take all (min (length all) (quotient most-elements (length first-row)))))

(define (twin-key f dt)
  (format "~a/~a" (fixture-name f) dt))

(define (sample-count f)
  (length (twin-rows f 'float64)))

(define (twin-ks f)
  (define rows (twin-rows f 'float64))
  (match-define (cons first-row _) rows)
  (define m (min (length rows) (length first-row)))
  (if (equal? (fixture-name f) "wide") (list 5 9 m) (list 1 3 m)))

(define (twin-case op f dt k #:whiten [whiten #f] #:solver [solver 'full] #:rows [rows 10])
  (match-define (cons first-row _) (twin-rows f dt))
  (hasheq 'op op
          'fixture (twin-key f dt)
          'd (length first-row)
          'k k
          'whiten whiten
          'dtype (symbol->string dt)
          'solver (symbol->string solver)
          'rows rows))

(define (twin-results cases)
  (define payload
    (for*/hasheq ([f (in-list fixtures)]
                  [dt (in-list '(float32 float64))])
      (values (string->symbol (twin-key f dt)) (append* (twin-rows f dt)))))
  (match-define (hash-table ('results results) ('version (hash-table ('cuml version))))
    (run-twin twin (hasheq 'cases cases 'fixtures payload)))
  (check-equal? version "26.08.00")
  (check-equal? version libcuml-version)
  results)

(struct ours (fitted noise n-samples transform))

(define (fit-ours f dt k #:whiten [whiten #f] #:solver [solver 'full])
  (define rows (twin-rows f dt))
  (match-define (cons first-row _) rows)
  (define X (rows->matrix rows dt))
  (define m (min (length rows) (length first-row)))
  (define result
    (pca-fit X #:n-components (and (< k m) k) #:whiten whiten #:solver solver #:tol 0.0))
  (ours (fitted-of result)
        (pca-result-noise-variance result)
        (pca-result-n-samples result)
        (device-matrix->list* (pca-transform result X))))

(define (unit dt)
  (if (eq? dt 'float32) (expt 2.0 -24) (expt 2.0 -53)))

(define (rows-of xs width)
  (for/list ([row (in-slice width xs)])
    row))

(define (sign x)
  (if (< x 0) -1.0 1.0))

(define (dot xs ys)
  (for/sum ([x (in-list xs)] [y (in-list ys)]) (* x y)))

(define (log2-ceiling n)
  (integer-length (sub1 n)))

(define (over-bound what got want bound)
  (and (> (abs (- got want)) bound)
       (format "~a: ~a vs ~a, bound ~a" what got want bound)))

(define (l14-problems mine theirs bounds dt solver #:whiten [whiten #f])
  (match-define (ours fitted noise n-samples transform) mine)
  (match-define (hash-table ('gaps gaps) ('total total) ('std std) ('row_norms row-norms)) bounds)
  (match-define (hash-table ('components c-flat)
                            ('explained_variance variance)
                            ('explained_variance_ratio ratio)
                            ('mean mean)
                            ('noise_variance theirs-noise)
                            ('n_samples theirs-n)
                            ('transform t-flat))
    theirs)
  (define u (unit dt))
  (define d (length mean))
  (define k (length variance))
  (match-define (cons largest _) (hash-ref bounds 'explained_variance))
  (define spread (* 64 u largest))
  (define components (rows-of c-flat d))
  (define signs
    (for/list ([c (in-list (fitted-components fitted))]
               [c-sk (in-list components)])
      (sign (dot c c-sk))))
  (define variance-bound (if (eq? solver 'jacobi) (* 8 d u largest) spread))
  (define n (length row-norms))
  (filter
   values
   (append
    (list (and (not (= n-samples theirs-n)) (format "n-samples ~a vs ~a" n-samples theirs-n))
          (over-bound "noise" noise theirs-noise spread))
    (for/list ([c (in-list (fitted-components fitted))]
               [c-sk (in-list components)]
               [s (in-list signs)]
               [gap (in-list gaps)]
               [i (in-naturals)])
      (over-bound (format "component ~a" i) (norm (map (lambda (a b) (- a (* s b))) c c-sk)) 0.0
                  (/ spread gap)))
    (for/list ([v (in-list (fitted-variance fitted))]
               [v-sk (in-list variance)]
               [i (in-naturals)])
      (over-bound (format "variance ~a" i) v v-sk variance-bound))
    (for/list ([r (in-list (fitted-ratio fitted))]
               [r-sk (in-list ratio)]
               [i (in-naturals)])
      (over-bound (format "ratio ~a" i) r r-sk (/ spread total)))
    (for/list ([m (in-list (fitted-mean fitted))]
               [m-sk (in-list mean)]
               [sd (in-list std)]
               [j (in-naturals)])
      (over-bound (format "mean ~a" j) m m-sk (* 8 u (log2-ceiling n) (+ (abs m-sk) sd))))
    (for*/list ([(row row-sk r) (in-parallel (in-list transform)
                                             (in-list (rows-of t-flat k))
                                             (in-list row-norms))]
                [(t t-sk s gap v-sk j)
                 (in-parallel (in-list row) (in-list row-sk) (in-list signs) (in-list gaps)
                              (in-list variance) (in-naturals))]
                #:unless (and whiten (< v-sk (* 1e-3 largest))))
      (define plain (* r (+ (/ spread gap) (* 4 d u))))
      (define bound
        (if whiten
            (+ (/ plain (sqrt v-sk)) (* (abs t-sk) (/ variance-bound (* 2 v-sk))))
            plain))
      (over-bound (format "transform column ~a" j) t (* s t-sk) bound)))))
