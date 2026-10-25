#lang racket/base

(require (only-in cuml/decomposition
                  pca-result-components
                  pca-result-explained-variance
                  pca-result-explained-variance-ratio
                  pca-result-mean
                  pca-result-singular-values)
         (only-in racket/list list-set)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in "scaler-data.rkt" generated-rows round-to rows->matrix))

(provide (struct-out fitted)
         (struct-out fixture)
         axis-rows
         column
         constant-column
         duplicate-rows
         first-max-entries
         fitted-of
         fixture-matrix
         fixture-rows
         fixtures
         gaps
         host-gram
         host-project
         host-reconstruct
         integer-fixture
         norm
         normal-fixture
         tie-rows
         wide-fixture)

(struct fixture (name source))

(struct fitted (components variance ratio singular mean) #:transparent)

(define (normal z _u _j)
  z)

(define (product a b)
  (define columns (apply map list b))
  (for/list ([row (in-list a)])
    (for/list ([c (in-list columns)])
      (for/sum ([x (in-list row)] [y (in-list c)]) (* x y)))))

(define (normals seed n d)
  (generated-rows seed n d 'float64 normal))

(define (correlated seed n d)
  (product (normals seed n d) (normals (add1 seed) d d)))

(define (shifted rows by)
  (for/list ([row (in-list rows)])
    (for/list ([x (in-list row)])
      (+ x by))))

(define (low-rank seed n d rank)
  (for/list ([row (in-list (product (normals seed n rank) (normals (add1 seed) rank d)))]
             [noise (in-list (normals (+ seed 2) n d))])
    (for/list ([x (in-list row)]
               [e (in-list noise)])
      (+ x (* 1e-3 e)))))

(define integer-fixture
  (fixture "uniform integers"
           (generated-rows 16 1797 64 'float64 (lambda (_z u _j) (floor (* 17 u))))))

(define wide-fixture (fixture "wide" (normals 18 50 200)))

(define normal-fixture (fixture "normal" (normals 11 2000 16)))

(define fixtures
  (list
   normal-fixture
   (fixture "correlated" (correlated 12 2000 16))
   (fixture "scales"
            (generated-rows 13 500 13 'float64 (lambda (z _u j) (* z (expt 10.0 (- (/ j 2.0) 3.0))))))
   (fixture "offset" (shifted (correlated 14 5000 8) 1e4))
   integer-fixture
   (fixture "low rank" (low-rank 17 1000 20 5))
   wide-fixture))

(define (fixture-rows f dtype)
  (define narrow (round-to dtype))
  (for/list ([row (in-list (fixture-source f))])
    (map narrow row)))

(define (fixture-matrix f dtype)
  (rows->matrix (fixture-rows f dtype) dtype))

(define (axis-rows a b c permutation)
  (define (on-axis axis value)
    (for/list ([column-axis (in-list permutation)])
      (if (= column-axis axis) value 0.0)))
  (append (for/list ([s (in-list '(1.0 -1.0 1.0 -1.0))])
            (on-axis 0 (* s a)))
          (for/list ([s (in-list '(1.0 -1.0))])
            (on-axis 1 (* s b)))
          (for/list ([s (in-list '(1.0 -1.0))])
            (on-axis 2 (* s c)))
          (list (on-axis -1 0.0))))

(define tie-rows '((1.0 -1.0) (-1.0 1.0) (2.0 -2.0) (-2.0 2.0) (0.5 0.5) (-0.5 -0.5)))

(define (constant-column n d j value)
  (for/list ([row (in-list (normals 19 n d))])
    (list-set row j value)))

(define (duplicate-rows n row)
  (for/list ([_ (in-range n)])
    row))

(define (fitted-of result)
  (fitted (device-matrix->list* (pca-result-components result))
          (device-vector->list (pca-result-explained-variance result))
          (device-vector->list (pca-result-explained-variance-ratio result))
          (device-vector->list (pca-result-singular-values result))
          (device-vector->list (pca-result-mean result))))

(define (dot xs ys)
  (for/sum ([x (in-list xs)] [y (in-list ys)]) (* x y)))

(define (norm xs)
  (sqrt (dot xs xs)))

(define (column rows j)
  (for/list ([row (in-list rows)])
    (list-ref row j)))

(define (centred row mean)
  (map - row mean))

(define (host-project rows mean components)
  (for/list ([row (in-list rows)])
    (define x (centred row mean))
    (for/list ([c (in-list components)])
      (dot x c))))

(define (host-reconstruct projections components mean)
  (define columns (apply map list components))
  (for/list ([t (in-list projections)])
    (for/list ([c (in-list columns)]
               [m (in-list mean)])
      (+ (dot t c) m))))

(define (host-gram components)
  (for/list ([a (in-list components)])
    (for/list ([b (in-list components)])
      (dot a b))))

(define (first-max-entries components)
  (for/list ([c (in-list components)])
    (for/fold ([best 0.0]) ([x (in-list c)])
      (if (> (abs x) (abs best)) x best))))

(define (gaps variances)
  (for/list ([v (in-list variances)]
             [i (in-naturals)])
    (for/fold ([gap +inf.0])
              ([w (in-list variances)]
               [j (in-naturals)]
               #:unless (= i j))
      (min gap (abs (- v w))))))
