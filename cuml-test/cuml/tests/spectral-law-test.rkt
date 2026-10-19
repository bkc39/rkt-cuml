#lang racket/base

(require (only-in racket/list range shuffle take)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/dbscan-data.rkt" mat pick same-partition? seeded)
         (only-in "private/spectral-data.rkt"
                  blobs
                  circles
                  graph
                  graph-cols
                  graph-labels
                  graph-map-vals
                  graph-n
                  graph-rows
                  graph-shuffle
                  graph-vals
                  knn-graph
                  labels
                  moons
                  permute-graph
                  permute-rows
                  planted?
                  sparse-blocks))

(struct planted (name rows truth k))

(define (dense-planted)
  (define (blob name n-each k d gap seed)
    (define-values (rows truth) (blobs n-each k d gap seed))
    (planted name rows truth k))
  (append (list (blob "blobs 3x100 at 50 sigma" 100 3 2 50.0 0))
          (for*/list ([n-each (in-list '(100 200))]
                      [seed (in-range 4)])
            (blob (format "blobs 4x~a at 20 sigma, seed ~a" n-each seed) n-each 4 3 20.0 seed))
          (list (blob "blobs 5x2000 at 8 sigma" 2000 5 8 8.0 7)
                (blob "blobs 5x2000 at 5 sigma" 2000 5 8 5.0 7))
          (let-values ([(rows truth) (circles 1000 0.5 0.03 0)]
                       [(moon-rows moon-truth) (moons 1000 0.05 0)])
            (list (planted "circles" rows truth 2) (planted "moons" moon-rows moon-truth 2)))))

(struct planted-graph (name graph truth k))

(define (graph-planted)
  (define (blocks name sizes p seed #:connected? [connected? #t])
    (define-values (g truth) (sparse-blocks sizes p seed #:connected? connected?))
    (planted-graph name g truth (length sizes)))
  (append
   (for*/list ([connected? (in-list '(#t #f))]
               [seed (in-range 6)])
     (blocks (format "blocks 30/40/50, ~a, seed ~a" (if connected? "connected" "disconnected") seed)
             '(30 40 50)
             0.3
             seed
             #:connected? connected?))
   (for/list ([seed (in-range 3)])
     (blocks (format "blocks 300/400/500/600, seed ~a" seed) '(300 400 500 600) 0.05 seed))))

(test-gpu "SC3: planted partitions are recovered, dense and as graphs"
  (for ([p (in-list (dense-planted))])
    (test-case (planted-name p)
      (check-true (planted? (labels (mat (planted-rows p) 'float32) #:n-clusters (planted-k p))
                            (planted-truth p)))))
  (for ([p (in-list (graph-planted))])
    (test-case (planted-graph-name p)
      (check-true (planted? (graph-labels (planted-graph-graph p) #:n-clusters (planted-graph-k p))
                            (planted-graph-truth p))))))

(define (small-blobs)
  (define-values (rows truth) (blobs 100 3 2 50.0 0))
  (values rows truth))

(define (blocks0)
  (define-values (g truth) (sparse-blocks '(30 40 50) 0.3 0))
  (values g truth))

(test-gpu "SC4: permuting rows or nodes permutes the partition; entry order changes nothing"
  (define-values (rows _truth) (small-blobs))
  (define-values (g _gtruth) (blocks0))
  (define base (labels (mat rows 'float32) #:n-clusters 3))
  (define graph-base (graph-labels g #:n-clusters 3))
  (for ([seed (in-range 3)])
    (test-case (format "seed ~a" seed)
      (define perm (seeded seed (lambda () (shuffle (range (length rows))))))
      (define permuted (labels (mat (permute-rows rows perm) 'float32) #:n-clusters 3))
      (check-true (same-partition? permuted (range 300) (pick base perm) (range 300)) "rows")
      (define node-perm (seeded seed (lambda () (shuffle (range (graph-n g))))))
      (check-true (same-partition? (graph-labels (permute-graph g node-perm) #:n-clusters 3)
                                   (range 120)
                                   (pick graph-base node-perm)
                                   (range 120))
                  "nodes")
      (check-equal? (graph-labels (graph-shuffle g seed) #:n-clusters 3) graph-base "entry order"))))

(test-gpu "SC5: weights scaled by powers of 4 give the same labels, by 1000 the same partition"
  (define-values (g _truth) (blocks0))
  (define base (graph-labels g #:n-clusters 3))
  (for ([factor (in-list '(4 16 0.25))])
    (check-equal? (graph-labels (graph-map-vals g (lambda (w) (* w factor))) #:n-clusters 3)
                  base
                  (format "x ~a" factor)))
  (check-true (same-partition? (graph-labels (graph-map-vals g (lambda (w) (* w 1000)))
                                             #:n-clusters 3)
                               (range 120)
                               base
                               (range 120))))

(test-gpu "SC6: the dense fit equals the fit of its kNN connectivity graph"
  (define-values (rows truth) (blobs 100 3 2 6.0 2))
  (for ([k (in-list '(10 15))])
    (test-case (format "~a neighbours" k)
      (define dense (labels (mat rows 'float32) #:n-clusters 3 #:n-neighbors k))
      (check-equal? (graph-labels (knn-graph rows k) #:n-clusters 3) dense)
      (check-true (planted? dense truth)))))

(test-gpu "SC7, SC8: halved entries and added zero weights change nothing"
  (define-values (g _truth) (blocks0))
  (define base (graph-labels g #:n-clusters 3))
  (define halves
    (graph (append (graph-rows g) (graph-rows g))
           (append (graph-cols g) (graph-cols g))
           (let ([half (map (lambda (w) (/ w 2)) (graph-vals g))]) (append half half))
           (graph-n g)))
  (check-equal? (graph-labels halves #:n-clusters 3) base "halves")
  (define zeros
    (graph (append (graph-rows g) '(0 5 7 119))
           (append (graph-cols g) '(5 0 119 7))
           (append (graph-vals g) '(0.0 0.0 0.0 0.0))
           (graph-n g)))
  (check-equal? (graph-labels zeros #:n-clusters 3) base "zeros"))

(test-gpu "SC9: isolated samples past the graph leave its partition unchanged"
  (define-values (g truth) (blocks0))
  (define wider (graph (graph-rows g) (graph-cols g) (graph-vals g) (+ (graph-n g) 10)))
  (define found (graph-labels wider #:n-clusters 3))
  (check-equal? (length found) 130)
  (check-true (planted? (take found 120) truth)))

(test-gpu "SC10: one seed gives identical labels in five fits; other seeds the same partition"
  (define-values (rows truth) (small-blobs))
  (define-values (g gtruth) (blocks0))
  (define-values (overlapping _otruth) (blobs 2000 5 8 3.0 7))
  (define X (mat rows 'float32))
  (define O (mat overlapping 'float32))
  (define cases
    (list (list "blobs" (lambda () (labels X #:n-clusters 3 #:random-state 5)))
          (list "graph" (lambda () (graph-labels g #:n-clusters 3 #:random-state 5)))
          (list "overlapping, one init" (lambda () (labels O #:n-clusters 5 #:n-init 1)))
          (list "overlapping, eigen-tol 1e-5"
                (lambda () (labels O #:n-clusters 5 #:eigen-tol 1e-5)))))
  (for ([c (in-list cases)])
    (test-case (car c)
      (define first-fit ((cadr c)))
      (for ([_ (in-range 4)])
        (check-equal? ((cadr c)) first-fit))))
  (for ([seed (in-list (list 0 1 12345 (expt 2 63)))])
    (test-case (format "seed ~a" seed)
      (check-true (planted? (labels X #:n-clusters 3 #:random-state seed) truth))
      (check-true (planted? (graph-labels g #:n-clusters 3 #:random-state seed) gtruth)))))
