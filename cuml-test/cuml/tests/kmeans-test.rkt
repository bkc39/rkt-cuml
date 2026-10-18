#lang racket/base

(require (only-in cuml/cluster
                  kmeans-fit
                  kmeans-predict
                  kmeans-result
                  kmeans-result-centroids
                  kmeans-result-labels
                  kmeans-score
                  kmeans-transform)
         (only-in racket/list make-list range)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft dtype shape)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/kmeans-data.rkt"
                  argmin-row
                  blob-data
                  cell-means
                  dtypes
                  far-start
                  first-rows
                  floats
                  grid64
                  grid64-means
                  grid64-start
                  grid8
                  grid8-labels
                  grid-weights
                  host
                  host-vector
                  mat
                  shift
                  squared-distance
                  start
                  step-weights
                  vec
                  weighted-grid8
                  weighted-start))

(define string-inits '(scalable-k-means++ k-means++ random))

(define (outcome r)
  (match-define (kmeans-result centroids labels inertia n-iter) r)
  (list (host centroids) (host-vector labels) inertia n-iter))

(define (fit-grid rows first dt . options)
  (apply kmeans-fit (mat rows dt) #:init (mat first dt) options))

(test-gpu "law A: every fit returns arrays of the documented shapes and dtypes"
  (for* ([dt (in-list dtypes)]
         [seed (in-range 1 6)])
    (define X (blob-data 300 3 4 seed dt))
    (for ([init (in-list (cons (first-rows X 4 dt) string-inits))])
      (test-case (format "~a, seed ~a, init ~a" dt seed init)
        (match-define (kmeans-result centroids labels inertia n-iter)
          (kmeans-fit X #:n-clusters 4 #:init init #:random-state seed #:max-iter 50))
        (check-equal? (list (shape centroids) (dtype centroids)) (list '(4 3) dt))
        (check-equal? (list (shape labels) (dtype labels)) '((300) int32))
        (check-true (for/and ([l (in-list (host-vector labels))])
                      (<= 0 l 3)))
        (define distances (kmeans-transform centroids X))
        (check-equal? (list (shape distances) (dtype distances)) (list '(300 4) dt))
        (check-true (for*/and ([row (in-list (host distances))]
                               [v (in-list row)])
                      (>= v 0.0)))
        (check-true (and (flonum? inertia) (>= inertia 0.0)))
        (check-true (<= 0 n-iter 50))))))

(define (norm2 v)
  (for/sum ([a (in-list v)])
    (* a a)))

(define (margin-ok? x row centroids tolerance)
  (match-define (list* best second _) (sort row <))
  (define scale (max 1.0 (+ (norm2 x) (apply max (map norm2 centroids)))))
  (> (- second best) (* tolerance scale)))

(test-gpu "law C: predict, score and transform agree with the fit that made the centroids"
  (for* ([dt (in-list dtypes)]
         [seed (in-range 1 9)])
    (test-case (format "~a, seed ~a" dt seed)
      (define X (blob-data 400 3 5 (+ seed 100) dt))
      (define w (vec (for/list ([i (in-range 400)]) (add1 (modulo (* i 7) 5))) dt))
      (match-define (kmeans-result centroids labels inertia _)
        (kmeans-fit X #:n-clusters 5 #:random-state seed))
      (check-equal? (host-vector (kmeans-predict centroids X)) (host-vector labels))
      (check-equal? (kmeans-score centroids X) (- inertia))
      (match-define (kmeans-result w-centroids _ w-inertia _)
        (kmeans-fit X #:n-clusters 5 #:random-state seed #:sample-weight w))
      (check-equal? (kmeans-score w-centroids X #:sample-weight w) (- w-inertia))
      (define init (first-rows X 5 dt))
      (define before (host init))
      (kmeans-fit X #:init init)
      (check-equal? (host init) before "an init matrix is not modified")
      (match-define (kmeans-result c0 l0 i0 n0) (kmeans-fit X #:init init #:max-iter 0))
      (check-equal? (list (host c0) n0) (list before 0))
      (check-equal? (host-vector l0) (host-vector (kmeans-predict init X)))
      (check-equal? i0 (- (kmeans-score init X)))
      (define rows (host X))
      (define cs (host centroids))
      (define tolerance (if (eq? dt 'float32) 1e-4 1e-12))
      (define predicted (host-vector (kmeans-predict centroids X)))
      (define-values (checked excluded)
        (for/fold ([checked 0]
                   [excluded 0])
                  ([x (in-list rows)]
                   [row (in-list (host (kmeans-transform centroids X)))]
                   [p (in-list predicted)])
          (cond
            [(margin-ok? x row cs tolerance)
             (check-equal? (argmin-row row) p)
             (values (add1 checked) excluded)]
            [else (values checked (add1 excluded))])))
      (check-true (< excluded (* 0.01 (+ checked excluded)))))))

(test-gpu "law C: the transform's argmin is predict on the grids, exactly"
  (for* ([dt (in-list dtypes)]
         [grid+start (in-list (list (list grid8 start) (list grid64 grid64-start)))])
    (match-define (list rows first) grid+start)
    (define X (mat rows dt))
    (define centroids (kmeans-result-centroids (kmeans-fit X #:init (mat first dt))))
    (check-equal? (map argmin-row (host (kmeans-transform centroids X)))
                  (host-vector (kmeans-predict centroids X)))))

(define (same-for-both-dtypes f)
  (match-define (list single double) (map f dtypes))
  (check-equal? single double "float32 and float64 agree bitwise"))

(test-gpu "law D: Lloyd steps on integer grids are exact"
  (same-for-both-dtypes
   (lambda (dt)
     (define one (outcome (fit-grid grid8 start dt #:max-iter 1)))
     (check-equal? one (list (floats cell-means) grid8-labels 16.0 1))
     (for ([m (in-list '(2 3))])
       (match-define (list _ _ _ n-iter) (outcome (fit-grid grid8 start dt #:max-iter m)))
       (check-equal? n-iter 2))
     (match-define (list fixed _ _ fixed-iter) (outcome (fit-grid grid8 cell-means dt)))
     (check-equal? (list fixed fixed-iter) (list (floats cell-means) 1))
     (match-define (list empty-cell _ _ _) (outcome (fit-grid grid8 far-start dt #:max-iter 1)))
     (check-equal? empty-cell (floats far-start))
     (match-define (list g64 _ _ g64-iter) (outcome (fit-grid grid64 grid64-start dt #:max-iter 1)))
     (check-equal? (list g64 g64-iter) (list (floats grid64-means) 1))
     one)))

(test-gpu "law D: weights are rescaled to sum to the sample count"
  (same-for-both-dtypes
   (lambda (dt)
     (define X (mat weighted-grid8 dt))
     (define w (vec grid-weights dt))
     (match-define (kmeans-result centroids _ inertia _)
       (kmeans-fit X #:init (mat weighted-start dt) #:sample-weight w))
     (check-equal? inertia 52.0)
     (check-equal? (kmeans-score centroids X #:sample-weight w) -52.0)
     (check-equal? (kmeans-score centroids X) -40.0)
     (check-equal? (kmeans-score centroids X #:sample-weight (vec (map (lambda (v) (* 2 v)) grid-weights) dt))
                   -52.0)
     (match-define (kmeans-result stepped _ _ _)
       (kmeans-fit X #:init (mat weighted-start dt) #:max-iter 1 #:sample-weight (vec step-weights dt)))
     (check-equal? (host stepped) '((1.0 1.5) (12.0 12.0)))
     (define unit (outcome (fit-grid grid8 start dt #:sample-weight (vec (make-list 8 1) dt))))
     (check-equal? unit (outcome (fit-grid grid8 start dt)))
     (check-equal? (outcome (fit-grid grid8 start dt #:sample-weight (vec (make-list 8 2) dt)))
                   unit)
     (list inertia (host stepped) unit))))

(test-gpu "law D: translating and permuting the grid moves the fit with it"
  (same-for-both-dtypes
   (lambda (dt)
     (match-define (list centroids labels _ _) (outcome (fit-grid grid64 grid64-start dt #:max-iter 1)))
     (define t '(5 -7))
     (match-define (list moved moved-labels _ _)
       (outcome (fit-grid (shift grid64 t) (shift grid64-start t) dt #:max-iter 1)))
     (check-equal? moved (shift centroids t))
     (check-equal? moved-labels labels)
     (define order (for/list ([i (in-range 64)]) (modulo (* i 13) 64)))
     (define permuted (for/list ([i (in-list order)]) (list-ref grid64 i)))
     (match-define (list p-centroids p-labels _ _)
       (outcome (fit-grid permuted grid64-start dt #:max-iter 1)))
     (check-equal? p-centroids centroids)
     (check-equal? p-labels (for/list ([i (in-list order)]) (list-ref labels i)))
     (list centroids labels))))

(test-gpu "law D: fits repeat for a seed, and an array init ignores n-init"
  (for ([dt (in-list dtypes)])
    (define X (mat grid64 dt))
    (for* ([init (in-list string-inits)]
           [seed (in-range 1 4)])
      (check-equal? (outcome (kmeans-fit X #:n-clusters 4 #:init init #:random-state seed))
                    (outcome (kmeans-fit X #:n-clusters 4 #:init init #:random-state seed))))
    (check-equal? (outcome (fit-grid grid64 grid64-start dt #:n-init 1))
                  (outcome (fit-grid grid64 grid64-start dt)))))

(test-gpu "law D: as many clusters as samples puts each sample in its own"
  (for ([dt (in-list dtypes)])
    (define X (mat grid8 dt))
    (match-define (kmeans-result centroids labels inertia _)
      (kmeans-fit X #:n-clusters 8 #:random-state 1))
    (check-equal? inertia 0.0)
    (check-equal? (sort (host-vector labels) <) (range 8))
    (check-equal? (kmeans-score centroids X) -0.0)))

(test-gpu "law D: the batch size changes nothing"
  (for ([dt (in-list dtypes)])
    (define X (mat grid64 dt))
    (define B (blob-data 500 4 6 3 dt))
    (define C (kmeans-result-centroids (kmeans-fit B #:n-clusters 6 #:random-state 2)))
    (define (on-batch b)
      (define centroids (mat grid64-means dt))
      (list (host-vector (kmeans-predict centroids X #:max-samples-per-batch b))
            (host (kmeans-transform centroids X #:max-samples-per-batch b))
            (outcome (fit-grid grid64 grid64-start dt #:max-samples-per-batch b))
            (host-vector (kmeans-predict C B #:max-samples-per-batch b))))
    (define reference (on-batch 32768))
    (for ([b (in-list '(0 1 7))])
      (check-equal? (on-batch b) reference (format "batch ~a" b)))))
