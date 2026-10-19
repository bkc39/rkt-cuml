#lang racket/base

(require (only-in compiler/find-exe find-exe)
         (only-in racket/file delete-directory/files make-temporary-directory)
         (only-in racket/list count range shuffle take)
         (only-in racket/match match match-define)
         (only-in racket/port port->string)
         (only-in rackunit check-equal? check-true fail test-case)
         (only-in raft/private/resource with-release)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/dbscan-data.rkt" mat pick same-partition? seeded)
         (only-in "private/spectral-data.rkt"
                  blobs
                  circles
                  fit-case
                  fit-case-data
                  fit-case-k
                  fit-case-name
                  fit-case-truth
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
                  sparse-blocks
                  three-blobs
                  three-blocks))

(define (planted name data truth k)
  (fit-case name data truth k (hasheq) 'labels))

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

(define (graph-planted)
  (define (blocks name sizes p seed #:connected? [connected? #t])
    (define-values (g truth) (sparse-blocks sizes p seed #:connected? connected?))
    (planted name g truth (length sizes)))
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
    (test-case (fit-case-name p)
      (check-true (planted? (labels (mat (fit-case-data p) 'float32) #:n-clusters (fit-case-k p))
                            (fit-case-truth p)))))
  (for ([p (in-list (graph-planted))])
    (test-case (fit-case-name p)
      (check-true (planted? (graph-labels (fit-case-data p) #:n-clusters (fit-case-k p))
                            (fit-case-truth p))))))

(test-gpu "SC4: permuting rows or nodes permutes the partition; entry order changes nothing"
  (define-values (rows _truth) (three-blobs))
  (define-values (g _gtruth) (three-blocks))
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
  (define-values (g _truth) (three-blocks))
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
  (define-values (g _truth) (three-blocks))
  (define base (graph-labels g #:n-clusters 3))
  (define halves
    (struct-copy graph
                 g
                 [rows (append (graph-rows g) (graph-rows g))]
                 [cols (append (graph-cols g) (graph-cols g))]
                 [vals (let ([half (map (lambda (w) (/ w 2)) (graph-vals g))]) (append half half))]))
  (check-equal? (graph-labels halves #:n-clusters 3) base "halves")
  (define zeros
    (struct-copy graph
                 g
                 [rows (append (graph-rows g) '(0 5 7 119))]
                 [cols (append (graph-cols g) '(5 0 119 7))]
                 [vals (append (graph-vals g) '(0.0 0.0 0.0 0.0))]))
  (check-equal? (graph-labels zeros #:n-clusters 3) base "zeros"))

(test-gpu "SC9: isolated samples past the graph leave its partition unchanged"
  (define-values (g truth) (three-blocks))
  (define wider (struct-copy graph g [n (+ (graph-n g) 10)]))
  (define found (graph-labels wider #:n-clusters 3))
  (check-equal? (length found) 130)
  (check-true (planted? (take found 120) truth)))

(define (changed a b)
  (count (lambda (x y) (not (= x y))) a b))

(test-gpu "SC10: one seed repeats its labels; other seeds give the same partition"
  (define-values (rows truth) (three-blobs))
  (define-values (g gtruth) (three-blocks))
  (define-values (overlapping _otruth) (blobs 2000 5 8 3.0 7))
  (define X (mat rows 'float32))
  (define O (mat overlapping 'float32))
  (define cases
    (list
     (list "blobs" 0 (lambda () (labels X #:n-clusters 3 #:random-state 5)))
     (list "graph" 0 (lambda () (graph-labels g #:n-clusters 3 #:random-state 5)))
     (list "overlapping, one init" 10 (lambda () (labels O #:n-clusters 5 #:n-init 1)))
     (list "overlapping, eigen-tol 1e-5" 10 (lambda () (labels O #:n-clusters 5 #:eigen-tol 1e-5)))))
  (for ([c (in-list cases)])
    (match-define (list name most fit) c)
    (test-case name
      (define first-fit (fit))
      (for ([_ (in-range 4)])
        (define again (fit))
        (if (zero? most)
            (check-equal? again first-fit)
            (check-true (<= (changed again first-fit) most))))))
  (for ([seed (in-list (list 0 1 12345 (expt 2 63)))])
    (test-case (format "seed ~a" seed)
      (check-true (planted? (labels X #:n-clusters 3 #:random-state seed) truth))
      (check-true (planted? (graph-labels g #:n-clusters 3 #:random-state seed) gtruth)))))

(define cross-process-program
  `(module cross-process racket/base
     (require (only-in cuml/cluster
                       spectral-clustering-fit
                       spectral-clustering-fit-graph
                       spectral-clustering-result-labels)
              (only-in cuml/tests/private/dbscan-data mat)
              (only-in cuml/tests/private/spectral-data
                       graph->vectors
                       graph-n
                       three-blobs
                       three-blocks)
              (only-in raft device-vector->list))
     (define-values (rows _truth) (three-blobs))
     (define-values (g _gtruth) (three-blocks))
     (define (labels-of r)
       (device-vector->list (spectral-clustering-result-labels r)))
     (write
      (list (labels-of (spectral-clustering-fit (mat rows 'float32) #:n-clusters 3 #:random-state 5))
            (labels-of (apply spectral-clustering-fit-graph
                              (append (graph->vectors g) (list (graph-n g)))
                              #:n-clusters 3
                              #:random-state 5))))))

(define (run-program program)
  (with-release ([dir
                  (make-temporary-directory "cuml-spectral-~a")
                  (lambda (dir) (delete-directory/files dir #:must-exist? #f))])
    (define path (build-path dir "program.rkt"))
    (call-with-output-file path (lambda (out) (write program out)))
    (define-values (process stdout stdin stderr) (subprocess #f #f #f (find-exe) path))
    (close-output-port stdin)
    (define errors (box ""))
    (define drain (thread (lambda () (set-box! errors (port->string stderr)))))
    (define output (port->string stdout))
    (thread-wait drain)
    (subprocess-wait process)
    (list (subprocess-status process) output (unbox errors))))

(define (run-program/timeout program seconds)
  (define custodian (make-custodian))
  (define done (make-channel))
  (parameterize ([current-custodian custodian]
                 [current-subprocess-custodian-mode 'kill])
    (thread (lambda () (channel-put done (run-program program)))))
  (begin0 (sync/timeout seconds done)
    (custodian-shutdown-all custodian)))

(define stderr-bytes (* 256 1024))

(test-case "SC10's runner returns when the child writes more than a pipe holds to stderr"
  (match (run-program/timeout `(module noisy racket/base
                                 (void (write-bytes (make-bytes ,stderr-bytes 120)
                                                    (current-error-port))))
                              20)
    [(list 0 "" errors) (check-equal? (string-length errors) stderr-bytes)]
    [#f (fail "the runner did not return within 20 s")]
    [(list status output errors)
     (fail (format "the child exited with ~a, ~a bytes out, ~a bytes err"
                   status
                   (string-length output)
                   (string-length errors)))]))

(test-gpu "SC10: one seed gives the same labels in another process"
  (define-values (rows _truth) (three-blobs))
  (define-values (g _gtruth) (three-blocks))
  (match (run-program cross-process-program)
    [(list 0 output _)
     (check-equal? (read (open-input-string output))
                   (list (labels (mat rows 'float32) #:n-clusters 3 #:random-state 5)
                         (graph-labels g #:n-clusters 3 #:random-state 5)))]
    [(list status _ errors) (fail (format "the fit's process exited with ~a:\n~a" status errors))]))
