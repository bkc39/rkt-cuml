#lang racket/base

(require (only-in cuml/decomposition tsvd-fit tsvd-inverse-transform tsvd-transform)
         (only-in cuml/decomposition/private/native libcuml-version)
         (only-in racket/list append* take)
         (only-in racket/match match match-define)
         racket/runtime-path
         (only-in rackunit check-equal?)
         (only-in raft device-matrix->list*)
         (only-in "pca-data.rkt" fixture fixture-name fixture-rows gaps)
         (only-in "python-env.rkt" run-twin)
         (only-in "scaler-data.rkt" generated-rows rows->matrix rows-of)
         (only-in "tsvd-data.rkt"
                  normal-fixture
                  normal-rows-of
                  offset-fixture
                  reference
                  svd
                  tie-fixture
                  tsvd-of))

(provide (struct-out twin-case)
         float-bits
         ours
         theirs-reference
         theirs-svd
         twin-cases
         twin-results)

(define-runtime-path twin "../python/tsvd_twin.py")

(struct twin-case (fixture k))

(define (product a b)
  (define columns (apply map list b))
  (for/list ([row (in-list a)])
    (for/list ([c (in-list columns)])
      (for/sum ([x (in-list row)] [y (in-list c)]) (* x y)))))

(define low-rank-fixture
  (fixture "low rank"
           (for/list ([row (in-list (product (normal-rows-of 51 800 5) (normal-rows-of 52 5 50)))]
                      [noise (in-list (normal-rows-of 53 800 50))])
             (for/list ([x (in-list row)]
                        [e (in-list noise)])
               (+ x (* 0.01 e))))))

(define uniform-fixture
  (fixture "uniform" (generated-rows 54 300 8 'float64 (lambda (_z u _j) (* 255.0 u)))))

(define twin-cases
  (list (twin-case normal-fixture 5)
        (twin-case low-rank-fixture 5)
        (twin-case offset-fixture 4)
        (twin-case uniform-fixture 7)
        (twin-case tie-fixture 1)))

(define (key f dt)
  (format "~a/~a" (fixture-name f) dt))

(define (width f)
  (match-define (cons first-row _) (fixture-rows f 'float64))
  (length first-row))

(define (twin-results requests)
  (define cases
    (for/list ([request (in-list requests)])
      (match-define (list op (twin-case f k) dt solver) request)
      (hasheq 'op op
              'fixture (key f dt)
              'd (width f)
              'k k
              'dtype (symbol->string dt)
              'solver (symbol->string solver))))
  (define payload
    (for*/hasheq ([c (in-list twin-cases)]
                  [dt (in-list '(float32 float64))])
      (define f (twin-case-fixture c))
      (values (string->symbol (key f dt)) (append* (fixture-rows f dt)))))
  (match-define (hash-table ('results results) ('version (hash-table ('cuml version))))
    (run-twin twin (hasheq 'cases cases 'fixtures payload)))
  (check-equal? version "26.08.00")
  (check-equal? version libcuml-version)
  results)

(define (theirs-svd theirs d)
  (match-define (hash-table ('components components)
                            ('explained_variance variance)
                            ('explained_variance_ratio ratio)
                            ('singular_values singular))
    theirs)
  (svd (rows-of components d) variance ratio singular))

(define (theirs-reference theirs exact d)
  (match-define (svd components variance ratio singular) (theirs-svd theirs d))
  (match-define (hash-table ('singular_values spectrum)) exact)
  (reference components
             variance
             ratio
             singular
             (take (gaps (map (lambda (s) (* s s)) spectrum)) (length singular))))

(define (float-bits dt xs)
  (define size (if (eq? dt 'float32) 4 8))
  (for/list ([x (in-list xs)])
    (integer-bytes->integer (real->floating-point-bytes x size) #f)))

(define (ours c dt solver)
  (match-define (twin-case f k) c)
  (define rows (fixture-rows f dt))
  (define X (rows->matrix rows dt))
  (define result (tsvd-fit X #:n-components k #:solver solver))
  (define T (tsvd-transform result X))
  (values rows
          (tsvd-of result)
          (device-matrix->list* T)
          (device-matrix->list* (tsvd-inverse-transform result T))))
