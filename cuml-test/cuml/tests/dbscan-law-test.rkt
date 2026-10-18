#lang racket/base

(require (only-in cuml/cluster dbscan-fit)
         (only-in racket/list range shuffle)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/dbscan-data.rkt"
                  cosine-cloud
                  cosine-within
                  dbscan-oracle
                  distance-matrix
                  euclidean-within
                  expected
                  expected-core
                  expected-labels
                  first-copies
                  integer-grid
                  mat
                  repeat-rows
                  result->lists
                  same-partition?
                  seeded
                  structure-problems
                  vec)
         (only-in "private/kmeans-data.rkt" dtypes))

(define fitted
  (make-keyword-procedure (lambda (kws vals X)
                            (result->lists (keyword-apply dbscan-fit kws vals (list X))))))

(define (oracle-of rows eps min-samples #:weights [weights #f])
  (dbscan-oracle (length rows)
                 (euclidean-within (list->vector rows) eps)
                 min-samples
                 #:weights weights))

(define (noise labels)
  (for/list ([l (in-list labels)]
             [i (in-naturals)]
             #:when (= l -1))
    i))

(define (adjacent? labels core e)
  (match-define (expected _ _ options) e)
  (for/and ([l (in-list labels)]
            [i (in-naturals)]
            #:unless (memv i core)
            #:unless (= l -1))
    (and (memv l (vector-ref options i)) #t)))

(define (check-batched found e)
  (match-define (list labels core) found)
  (match-define (expected want-labels want-core _) e)
  (check-equal? (structure-problems labels core) '())
  (check-equal? core want-core "core samples")
  (check-equal? (map (lambda (i) (list-ref labels i)) core)
                (map (lambda (i) (list-ref want-labels i)) want-core)
                "core labels")
  (check-equal? (noise labels) (noise want-labels) "noise")
  (check-true (adjacent? labels core e) "border labels are a core neighbour's"))

(define (check-one-batch found e)
  (check-batched found e)
  (check-equal? found (list (expected-labels e) (expected-core e)) "the smallest neighbouring label"))

(test-gpu "L2: random integer grids match the host oracle, border points taking the smallest label"
  (for* ([dt (in-list dtypes)]
         [seed (in-range 1 6)])
    (test-case (format "~a seed ~a" dt seed)
      (define rows (integer-grid 150 2 12 seed))
      (check-one-batch (fitted (mat rows dt) #:eps 1.5 #:min-samples 4) (oracle-of rows 3/2 4)))))

(define big-rows (integer-grid 2001 2 60 21))
(define big-oracle (oracle-of big-rows 3/2 4))

(test-gpu "L3: the core samples, their clusters and the noise do not depend on the batch size"
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (define X (mat big-rows dt))
      (for ([mbytes (in-list '(1 2 5))])
        (check-batched (fitted X #:eps 1.5 #:min-samples 4 #:max-mbytes-per-batch mbytes) big-oracle))
      (check-one-batch (fitted X #:eps 1.5 #:min-samples 4) big-oracle)
      (check-equal? (fitted X #:eps 1.5 #:min-samples 4 #:max-mbytes-per-batch 0)
                    (fitted X #:eps 1.5 #:min-samples 4)))))

(test-gpu "L4: permuting the rows permutes the core samples and the noise"
  (define rows (integer-grid 300 2 20 8))
  (define p (seeded 3 (lambda () (shuffle (range 300)))))
  (define permuted (map (lambda (i) (list-ref rows i)) p))
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (match-define (list labels core) (fitted (mat rows dt) #:eps 1.5 #:min-samples 4))
      (match-define (list p-labels p-core) (fitted (mat permuted dt) #:eps 1.5 #:min-samples 4))
      (define (mapped indices)
        (for/list ([j (in-range 300)]
                   #:when (memv (list-ref p j) indices))
          j))
      (check-equal? p-core (mapped core))
      (check-equal? (noise p-labels) (mapped (noise labels)))
      (define p-mapped (map (lambda (j) (list-ref labels (list-ref p j))) (range 300)))
      (check-true (same-partition? p-labels p-core p-mapped p-core))
      (check-true (adjacent? p-labels p-core (oracle-of permuted 3/2 4))))))

(test-gpu "L5: translating or negating an integer grid changes nothing"
  (define rows (integer-grid 300 3 10 12))
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (define found (fitted (mat rows dt) #:eps 1.5 #:min-samples 5))
      (check-equal?
       (fitted (mat (map (lambda (r) (map + r '(-7 40 3))) rows) dt) #:eps 1.5 #:min-samples 5)
       found)
      (check-equal? (fitted (mat (map (lambda (r) (map - r)) rows) dt) #:eps 1.5 #:min-samples 5)
                    found))))

(test-gpu "L6: integer weights act as repeated rows; negative weights subtract"
  (define rows (integer-grid 200 2 16 5))
  (define weights (seeded 9 (lambda () (map (lambda (_) (add1 (random 4))) rows))))
  (define repeated (repeat-rows rows weights))
  (define firsts (first-copies weights))
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (match-define (list labels core)
        (fitted (mat rows dt) #:eps 1.5 #:min-samples 7 #:sample-weight (vec weights dt)))
      (match-define (list r-labels r-core) (fitted (mat repeated dt) #:eps 1.5 #:min-samples 7))
      (check-equal? labels (map (lambda (i) (list-ref r-labels i)) firsts))
      (check-equal? core
                    (for/list ([f (in-list firsts)]
                               [i (in-naturals)]
                               #:when (memv f r-core))
                      i))
      (match-define (expected w-labels w-core _) (oracle-of rows 3/2 7 #:weights weights))
      (check-equal? (list labels core) (list w-labels w-core))
      (check-equal? (fitted (mat rows dt)
                            #:eps 1.5
                            #:min-samples 4
                            #:sample-weight (vec (map (lambda (_) 1) rows) dt))
                    (fitted (mat rows dt) #:eps 1.5 #:min-samples 4))
      (check-equal?
       (fitted (mat '((0) (10) (20)) dt) #:eps 1 #:min-samples 5 #:sample-weight (vec '(1 5 1) dt))
       '((-1 0 -1) (1)))
      (define negative '((1) (1) (0) (2)))
      (check-equal?
       (fitted (mat negative dt) #:eps 1 #:min-samples 5 #:sample-weight (vec '(1 1 3 -1) dt))
       '((0 0 0 -1) (2)))
      (check-equal? (fitted (mat negative dt) #:eps 1 #:min-samples 5) '((-1 -1 -1 -1) ()))
      (check-equal? (fitted (mat negative dt) #:eps 1 #:min-samples 4) '((0 0 0 0) (0 1)))
      (check-equal?
       (fitted (mat negative dt) #:eps 1 #:min-samples 5 #:sample-weight (vec '(1 1 3 1) dt))
       '((0 0 0 0) (0 1 2))))))

(define pre-rows (integer-grid 2001 2 60 33))
(define pre-distances (distance-matrix pre-rows))

(test-gpu "L7: precomputed distances give the euclidean labels, at one batch and at many"
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (define X (mat pre-rows dt))
      (define D (mat pre-distances dt))
      (for* ([eps (in-list '(1.5 2.5))]
             [mbytes (in-list '(#f 1))])
        (check-equal?
         (fitted D #:eps eps #:min-samples 5 #:metric 'precomputed #:max-mbytes-per-batch mbytes)
         (fitted X #:eps eps #:min-samples 5 #:max-mbytes-per-batch mbytes)
         (format "eps ~a, max-mbytes-per-batch ~a" eps mbytes))))))

(test-gpu "L8: cosine labels match the host oracle on one minus the cosine similarity"
  (define eps 0.02)
  (define rows (cosine-cloud 300 3 eps 4))
  (define e (dbscan-oracle 300 (cosine-within (list->vector rows) eps) 4))
  (check-true (pair? (expected-core e)))
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (check-one-batch (fitted (mat rows dt) #:eps eps #:min-samples 4 #:metric 'cosine) e))))
