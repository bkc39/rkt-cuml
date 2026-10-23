#lang racket/base

(require (only-in racket/flonum flsingle)
         (only-in racket/list make-list range shuffle)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/dbscan-data.rkt"
                  cosine-cloud
                  cosine-distance
                  cosine-within
                  dbscan-oracle
                  distance-matrix
                  euclidean-within
                  expected
                  expected-core
                  expected-labels
                  first-copies
                  fitted-lists
                  integer-grid
                  mat
                  pick
                  repeat-rows
                  same-partition?
                  seeded
                  structure-problems
                  vec)
         (only-in "private/kmeans-data.rkt" dtypes))

(define (oracle-of rows eps min-samples #:weights [weights #f])
  (dbscan-oracle (length rows)
                 (euclidean-within (list->vector rows) eps)
                 min-samples
                 #:weights weights))

(define (wide-pair d)
  (define tenth (flsingle 0.1))
  (define half (quotient d 2))
  (list (cons 1.0 (make-list (- d 1) tenth))
        (cons 1.0 (append (make-list half (- tenth)) (make-list (- d 1 half) tenth)))))

(define (wide-pair-fit d factor)
  (match-define (list a b) (wide-pair d))
  (define eps (* factor (cosine-distance a b)))
  (values (fitted-lists (mat (list a b) 'float32) #:eps eps #:min-samples 2 #:metric 'cosine)
          (dbscan-oracle 2 (cosine-within (vector a b) eps) 2)))

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
  (check-equal? (pick labels core) (pick want-labels want-core) "core labels")
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
      (check-one-batch (fitted-lists (mat rows dt) #:eps 1.5 #:min-samples 4)
                       (oracle-of rows 3/2 4)))))

(define big-rows (integer-grid 2001 2 60 21))
(define big-oracle (oracle-of big-rows 3/2 4))

(test-gpu "L3: the core samples, their clusters and the noise do not depend on the batch size"
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (define X (mat big-rows dt))
      (for ([mbytes (in-list '(1 2 5))])
        (check-batched (fitted-lists X #:eps 1.5 #:min-samples 4 #:max-mbytes-per-batch mbytes)
                       big-oracle))
      (check-one-batch (fitted-lists X #:eps 1.5 #:min-samples 4) big-oracle)
      (check-equal? (fitted-lists X #:eps 1.5 #:min-samples 4 #:max-mbytes-per-batch 0)
                    (fitted-lists X #:eps 1.5 #:min-samples 4)))))

(test-gpu "L4: permuting the rows permutes the core samples and the noise"
  (define rows (integer-grid 300 2 20 8))
  (define p (seeded 3 (lambda () (shuffle (range 300)))))
  (define permuted (pick rows p))
  (define at (list->vector p))
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (match-define (list labels core) (fitted-lists (mat rows dt) #:eps 1.5 #:min-samples 4))
      (match-define (list p-labels p-core) (fitted-lists (mat permuted dt) #:eps 1.5 #:min-samples 4))
      (define (mapped indices)
        (define chosen (make-vector 300 #f))
        (for ([i (in-list indices)])
          (vector-set! chosen i #t))
        (for/list ([j (in-range 300)]
                   #:when (vector-ref chosen (vector-ref at j)))
          j))
      (check-equal? p-core (mapped core))
      (check-equal? (noise p-labels) (mapped (noise labels)))
      (define p-mapped (pick labels p))
      (check-true (same-partition? p-labels p-core p-mapped p-core))
      (check-true (adjacent? p-labels p-core (oracle-of permuted 3/2 4))))))

(test-gpu "L5: translating or negating an integer grid changes nothing"
  (define rows (integer-grid 300 3 10 12))
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (define found (fitted-lists (mat rows dt) #:eps 1.5 #:min-samples 5))
      (check-equal?
       (fitted-lists (mat (map (lambda (r) (map + r '(-7 40 3))) rows) dt) #:eps 1.5 #:min-samples 5)
       found)
      (check-equal?
       (fitted-lists (mat (map (lambda (r) (map - r)) rows) dt) #:eps 1.5 #:min-samples 5)
       found))))

(test-gpu "L6: integer weights act as repeated rows; negative weights subtract"
  (define rows (integer-grid 200 2 16 5))
  (define weights (seeded 9 (lambda () (map (lambda (_) (add1 (random 4))) rows))))
  (define repeated (repeat-rows rows weights))
  (define firsts (first-copies weights))
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (match-define (list labels core)
        (fitted-lists (mat rows dt) #:eps 1.5 #:min-samples 7 #:sample-weight (vec weights dt)))
      (match-define (list r-labels r-core) (fitted-lists (mat repeated dt) #:eps 1.5 #:min-samples 7))
      (check-equal? labels (pick r-labels firsts))
      (check-equal? core
                    (for/list ([f (in-list firsts)]
                               [i (in-naturals)]
                               #:when (memv f r-core))
                      i))
      (match-define (expected w-labels w-core _) (oracle-of rows 3/2 7 #:weights weights))
      (check-equal? (list labels core) (list w-labels w-core))
      (check-equal? (fitted-lists (mat rows dt)
                                  #:eps 1.5
                                  #:min-samples 4
                                  #:sample-weight (vec (map (lambda (_) 1) rows) dt))
                    (fitted-lists (mat rows dt) #:eps 1.5 #:min-samples 4))
      (check-equal? (fitted-lists (mat '((0) (10) (20)) dt)
                                  #:eps 1
                                  #:min-samples 5
                                  #:sample-weight (vec '(1 5 1) dt))
                    '((-1 0 -1) (1)))
      (define negative '((1) (1) (0) (2)))
      (check-equal?
       (fitted-lists (mat negative dt) #:eps 1 #:min-samples 5 #:sample-weight (vec '(1 1 3 -1) dt))
       '((0 0 0 -1) (2)))
      (check-equal? (fitted-lists (mat negative dt) #:eps 1 #:min-samples 5) '((-1 -1 -1 -1) ()))
      (check-equal? (fitted-lists (mat negative dt) #:eps 1 #:min-samples 4) '((0 0 0 0) (0 1)))
      (check-equal?
       (fitted-lists (mat negative dt) #:eps 1 #:min-samples 5 #:sample-weight (vec '(1 1 3 1) dt))
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
        (check-equal? (fitted-lists D
                                    #:eps eps
                                    #:min-samples 5
                                    #:metric 'precomputed
                                    #:max-mbytes-per-batch mbytes)
                      (fitted-lists X #:eps eps #:min-samples 5 #:max-mbytes-per-batch mbytes)
                      (format "eps ~a, max-mbytes-per-batch ~a" eps mbytes))))))

(test-gpu "L8: cosine labels match the host oracle on one minus the cosine similarity"
  (define eps 0.02)
  (define rows (cosine-cloud 300 3 eps 4))
  (define e (dbscan-oracle 300 (cosine-within (list->vector rows) eps) 4))
  (check-true (pair? (expected-core e)))
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (check-one-batch (fitted-lists (mat rows dt) #:eps eps #:min-samples 4 #:metric 'cosine) e))))

(test-gpu "L8: scaling each row by a factor in [0.1, 10] leaves the cosine labels and the oracle's"
  (define eps 0.02)
  (define rows (cosine-cloud 300 3 eps 4))
  (define scaled
    (seeded 9
            (lambda ()
              (for/list ([row (in-list rows)])
                (define factor (/ (+ 10 (random 991)) 100))
                (for/list ([x (in-list row)])
                  (* factor x))))))
  (define e (dbscan-oracle 300 (cosine-within (list->vector scaled) eps) 4))
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (define found (fitted-lists (mat scaled dt) #:eps eps #:min-samples 4 #:metric 'cosine))
      (check-one-batch found e)
      (check-equal? found (fitted-lists (mat rows dt) #:eps eps #:min-samples 4 #:metric 'cosine)))))

(test-gpu "L8: cosine labels do not depend on the order or the scale of the rows"
  (for* ([dt (in-list dtypes)]
         [c (in-list '((((2 0) (0 1) (4 0)) (0 -1 0)) (((0 1) (2 0) (4 0)) (-1 0 0))))])
    (match-define (list rows labels) c)
    (test-case (format "~a ~a" dt rows)
      (match-define (list found _)
        (fitted-lists (mat rows dt) #:eps 0.49 #:min-samples 2 #:metric 'cosine))
      (check-equal? found labels))))

(test-gpu "L8: rows along one direction are cosine neighbours at any eps"
  (for* ([dt (in-list dtypes)]
         [eps (in-list '(1e-30 1e-6 0.49))])
    (test-case (format "~a eps ~a" dt eps)
      (check-equal? (fitted-lists (mat '((1 0) (2 0)) dt) #:eps eps #:min-samples 2 #:metric 'cosine)
                    '((0 0) (0 1))))))

(test-gpu "L8: a pair at cosine distance exactly eps is a neighbour pair"
  (for* ([dt (in-list dtypes)]
         [c (in-list '((((1 0) (0 1)) 1) (((1 1 1 1) (1 -1 -1 -1)) 1.5)
                                         (((1 1 1 1) (1 1 1 -1)) 0.5)
                                         (((1 0) (-1 0)) 2)))])
    (match-define (list rows eps) c)
    (test-case (format "~a ~a eps ~a" dt rows eps)
      (check-equal? (fitted-lists (mat rows dt) #:eps eps #:min-samples 2 #:metric 'cosine)
                    '((0 0) (0 1))))))

(test-gpu "L8: at eps 2 every pair is a neighbour pair"
  (for* ([dt (in-list dtypes)]
         [rows (in-list '(((8 6) (-8 -6)) ((24 10) (-24 -10))))])
    (test-case (format "~a ~a" dt rows)
      (check-equal? (fitted-lists (mat rows dt) #:eps 2 #:min-samples 2 #:metric 'cosine)
                    '((0 0) (0 1))))))

(test-gpu "L8: float32 cosine labels match the oracle at eps 1e-3, and at 1024 columns 1e-5, either side of a wide pair's distance"
  (for ([c (in-list `((1024 0.999) (1024 1.001)
                                   (1024 ,(- 1 1e-5))
                                   (1024 ,(+ 1 1e-5))
                                   (4096 0.999)
                                   (4096 1.001)))])
    (match-define (list d factor) c)
    (test-case (format "d ~a eps ~a t" d factor)
      (define-values (found e) (wide-pair-fit d factor))
      (check-equal? found (list (expected-labels e) (expected-core e))))))

(test-gpu "L8: float32 cosine at 4096 columns stays within 3e-5 of eps (double row sums)"
  (define-values (found _) (wide-pair-fit 4096 (+ 1 3e-5)))
  (check-equal? found '((0 0) (0 1))))

(test-gpu "L8: a subnormal eps under cosine returns"
  (check-equal?
   (fitted-lists (mat '((1 0) (0 1)) 'float64) #:eps 5e-324 #:min-samples 2 #:metric 'cosine)
   '((-1 -1) ())))
