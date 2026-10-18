#lang racket/base

(require (only-in cuml/cluster dbscan-fit dbscan-result)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in raft device-matrix->list* dtype shape)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/dbscan-data.rkt"
                  distance-matrix
                  grid-core
                  grid-labels
                  grid-rows
                  grid-weighted-core
                  grid-weights
                  integer-grid
                  mat
                  result->lists
                  seeded
                  structure-problems
                  vec)
         (only-in "private/kmeans-data.rkt" dtypes))

(define out-dtypes '(int32 int64))

(define fitted
  (make-keyword-procedure (lambda (kws vals X)
                            (result->lists (keyword-apply dbscan-fit kws vals (list X))))))

(define (fit-lists X #:eps eps #:min-samples min-samples #:metric [metric 'euclidean])
  (define lists (result->lists (dbscan-fit X #:eps eps #:min-samples min-samples #:metric metric)))
  (check-equal? (apply structure-problems lists) '())
  lists)

(test-gpu "L1: the hand grid's labels and core samples, at both dtypes and both out-dtypes"
  (for* ([dt (in-list dtypes)]
         [out (in-list out-dtypes)])
    (test-case (format "~a ~a" dt out)
      (define X (mat grid-rows dt))
      (match-define (dbscan-result labels core)
        (dbscan-fit X #:eps 2 #:min-samples 4 #:out-dtype out))
      (check-equal? (result->lists (dbscan-result labels core)) (list grid-labels grid-core))
      (check-equal? (list (shape labels) (dtype labels) (shape core) (dtype core))
                    (list '(8) out '(2) out))
      (check-equal? (cadr (result->lists (dbscan-fit X
                                                     #:eps 2
                                                     #:min-samples 4
                                                     #:out-dtype out
                                                     #:sample-weight (vec grid-weights dt))))
                    grid-weighted-core)
      (check-equal?
       (fit-lists (mat (distance-matrix grid-rows) dt) #:eps 2 #:min-samples 4 #:metric 'precomputed)
       (list grid-labels grid-core)))))

(define (unround-cloud n d seed)
  (seeded seed
          (lambda ()
            (for/list ([_ (in-range n)])
              (for/list ([_ (in-range d)])
                (+ 0.1 (/ (random 1000) 7.0)))))))

(test-gpu "L9: a fit leaves X bit-identical under every metric"
  (define rows (unround-cloud 300 3 5))
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (for ([metric (in-list '(euclidean cosine))]
            [eps (in-list '(20 0.01))])
        (define X (mat rows dt))
        (define before (device-matrix->list* X))
        (dbscan-fit X #:eps eps #:min-samples 3 #:metric metric #:max-mbytes-per-batch 1)
        (dbscan-fit X #:eps eps #:min-samples 3 #:metric metric)
        (check-equal? (device-matrix->list* X) before (format "~a" metric)))
      (define D (mat (distance-matrix (integer-grid 200 2 30 3)) dt))
      (define before (device-matrix->list* D))
      (dbscan-fit D #:eps 2.5 #:min-samples 3 #:metric 'precomputed)
      (check-equal? (device-matrix->list* D) before "precomputed"))))

(test-gpu "L10: int64 labels hold int32's values"
  (define rows (integer-grid 500 2 25 9))
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (define X (mat rows dt))
      (match-define (dbscan-result labels core)
        (dbscan-fit X #:eps 1.5 #:min-samples 4 #:out-dtype 'int64))
      (check-equal? (list (dtype labels) (dtype core)) '(int64 int64))
      (check-equal? (result->lists (dbscan-result labels core))
                    (fit-lists X #:eps 1.5 #:min-samples 4)))))

(test-gpu "L11: a distance equal to eps is a neighbour"
  (for ([dt (in-list dtypes)])
    (check-equal? (fit-lists (mat '((0) (1) (2) (3) (4)) dt) #:eps 1 #:min-samples 3)
                  '((0 0 0 0 0) (1 2 3)))))

(test-gpu "L12: min-samples 1 makes every row core, and one row is its own cluster or noise"
  (for* ([dt (in-list dtypes)]
         [out (in-list out-dtypes)])
    (test-case (format "~a ~a" dt out)
      (check-equal? (fitted (mat '((0) (1) (5) (6) (10)) dt) #:eps 1 #:min-samples 1 #:out-dtype out)
                    '((0 0 1 1 2) (0 1 2 3 4)))
      (define one (mat '((3)) dt))
      (check-equal? (fitted one #:min-samples 1 #:out-dtype out) '((0) (0)))
      (match-define (dbscan-result labels core) (dbscan-fit one #:min-samples 2 #:out-dtype out))
      (check-equal? (result->lists (dbscan-result labels core)) '((-1) ()))
      (check-equal? (list (shape core) (dtype core)) (list '(0) out)))))

(define (insert-at rows k row)
  (append (for/list ([r (in-list rows)]
                     [_ (in-range k)])
            r)
          (list row)
          (list-tail rows k)))

(define (without labels core k)
  (list (append (for/list ([l (in-list labels)]
                           [_ (in-range k)])
                  l)
                (list-tail labels (add1 k)))
        (for/list ([i (in-list core)])
          (if (> i k)
              (sub1 i)
              i))))

(define (precomputed-with rows k bad)
  (define D (distance-matrix rows))
  (for/list ([row (in-list (insert-at D k (make-list (length rows) bad)))])
    (insert-at row k bad)))

(define (check-isolated clean dirty k)
  (match-define (list labels core) dirty)
  (check-equal? (list-ref labels k) -1)
  (check-equal? (without labels core k) clean))

(test-gpu "L13: a NaN or infinite row is noise and leaves the other labels alone"
  (define rows (map (lambda (r) (map add1 r)) (integer-grid 80 2 8 4)))
  (define k 7)
  (for* ([dt (in-list dtypes)]
         [metric (in-list '(euclidean cosine))]
         [bad (in-list '(+nan.0 +inf.0 -inf.0))])
    (test-case (format "~a ~a ~a" dt metric bad)
      (define eps (if (eq? metric 'cosine) 0.001 1.5))
      (check-isolated
       (fit-lists (mat rows dt) #:eps eps #:min-samples 3 #:metric metric)
       (fitted (mat (insert-at rows k (list bad 1)) dt) #:eps eps #:min-samples 3 #:metric metric)
       k)))
  (for* ([dt (in-list dtypes)]
         [bad (in-list '(+nan.0 +inf.0))])
    (test-case (format "~a precomputed ~a" dt bad)
      (check-isolated
       (fit-lists (mat (distance-matrix rows) dt) #:eps 1.5 #:min-samples 3 #:metric 'precomputed)
       (fitted (mat (precomputed-with rows k bad) dt) #:eps 1.5 #:min-samples 3 #:metric 'precomputed)
       k))))

(test-gpu "L14: core indices ascend, labels run from -1 to k - 1, numbered by smallest core index"
  (for ([seed (in-range 1 6)])
    (define X (mat (integer-grid 400 2 30 seed) 'float32))
    (for ([eps (in-list '(1 1.5 2.5))]
          [min-samples (in-list '(2 4 7))])
      (match-define (list labels _) (fit-lists X #:eps eps #:min-samples min-samples))
      (check-equal? (length labels) 400)
      (check-equal? (fitted X #:eps eps #:min-samples 401) (list (make-list 400 -1) '())))))
