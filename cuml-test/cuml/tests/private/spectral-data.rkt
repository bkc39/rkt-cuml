#lang racket/base

(require (only-in cuml/cluster
                  spectral-clustering-fit
                  spectral-clustering-fit-graph
                  spectral-clustering-result-labels)
         (only-in racket/flonum flsingle)
         (only-in racket/list append* make-list range take)
         (only-in racket/match match-lambda)
         (only-in racket/math pi sqr)
         (only-in raft device-vector->list list->device-vector)
         (only-in threading ~>)
         (only-in "dbscan-data.rkt" same-partition? seeded))

(provide (struct-out graph)
         blobs
         circles
         cliques
         float32-rows
         graph->vectors
         graph-dense
         graph-labels
         graph-map-vals
         graph-n
         graph-scale-upper
         graph-shuffle
         graph-upper
         knn-graph
         labels
         moons
         permute-graph
         permute-rows
         planted?
         repeat-each
         sparse-blocks
         truth-of)

(struct graph (rows cols vals n) #:transparent)

(define (float32-rows rows)
  (for/list ([r (in-list rows)])
    (map (lambda (x) (flsingle (exact->inexact x))) r)))

(define (planted? labels truth)
  (define all (range (length truth)))
  (same-partition? labels all truth all))

(define (gaussian)
  (* (sqrt (* -2.0 (log (- 1.0 (random))))) (cos (* 2.0 pi (random)))))

(define (truth-of sizes)
  (append* (for/list ([s (in-list sizes)]
                      [g (in-naturals)])
             (make-list s g))))

(define (blobs n-each k d gap seed #:sd [sd 1.0])
  (seeded seed
          (lambda ()
            (values (float32-rows (for*/list ([g (in-range k)]
                                              [_ (in-range n-each)])
                                    (for/list ([_ (in-range d)])
                                      (+ (* g gap) (* sd (gaussian))))))
                    (truth-of (make-list k n-each))))))

(define (linspace start stop count #:endpoint? [endpoint? #t])
  (define steps
    (if endpoint?
        (max 1 (sub1 count))
        count))
  (for/list ([i (in-range count)])
    (+ start (* i (/ (- stop start) steps)))))

(define (noisy points noise)
  (float32-rows (for/list ([p (in-list points)])
                  (for/list ([x (in-list p)])
                    (+ x (* noise (gaussian)))))))

(define (circles n factor noise seed)
  (define n-out (quotient n 2))
  (define n-in (- n n-out))
  (define (ring count scale)
    (for/list ([t (in-list (linspace 0.0 (* 2 pi) count #:endpoint? #f))])
      (list (* scale (cos t)) (* scale (sin t)))))
  (seeded seed
          (lambda ()
            (values (noisy (append (ring n-out 1.0) (ring n-in factor)) noise)
                    (truth-of (list n-out n-in))))))

(define (moons n noise seed)
  (define n-out (quotient n 2))
  (define n-in (- n n-out))
  (define outer
    (for/list ([t (in-list (linspace 0.0 pi n-out))])
      (list (cos t) (sin t))))
  (define inner
    (for/list ([t (in-list (linspace 0.0 pi n-in))])
      (list (- 1 (cos t)) (- 1 (sin t) 0.5))))
  (seeded seed (lambda () (values (noisy (append outer inner) noise) (truth-of (list n-out n-in))))))

(define (edges->graph edges n)
  (define ordered
    (sort (hash-keys edges)
          (lambda (a b) (or (< (car a) (car b)) (and (= (car a) (car b)) (< (cdr a) (cdr b)))))))
  (graph (map car ordered) (map cdr ordered) (map (lambda (e) (hash-ref edges e)) ordered) n))

(define (set-edge! edges i j w)
  (hash-set! edges (cons i j) w)
  (hash-set! edges (cons j i) w))

(define (sparse-blocks sizes p seed #:between [between 2] #:connected? [connected? #t])
  (define starts
    (for/fold ([at 0]
               [starts '()]
               #:result (reverse starts))
              ([s (in-list sizes)])
      (values (+ at s) (cons at starts))))
  (define edges (make-hash))
  (seeded seed
          (lambda ()
            (for ([s (in-list sizes)]
                  [o (in-list starts)])
              (for* ([i (in-range s)]
                     [j (in-range (add1 i) s)])
                (define keep? (< (random) p))
                (define w (+ 0.5 (* 0.5 (random))))
                (when keep?
                  (set-edge! edges (+ o i) (+ o j) w)))
              (for ([i (in-range (sub1 s))])
                (define at (cons (+ o i) (+ o i 1)))
                (set-edge! edges (+ o i) (+ o i 1) (max 0.5 (hash-ref edges at 0.0)))))
            (when connected?
              (for ([a (in-list sizes)]
                    [b (in-list (cdr sizes))]
                    [oa (in-list starts)]
                    [ob (in-list (cdr starts))])
                (for ([_ (in-range between)])
                  (set-edge! edges (+ oa (random a)) (+ ob (random b)) 0.05))))))
  (values (edges->graph edges (apply + sizes)) (truth-of sizes)))

(define (squared-distance a b)
  (for/sum ([x (in-list a)] [y (in-list b)]) (sqr (- x y))))

(define (knn-graph rows k)
  (define points (list->vector rows))
  (define n (vector-length points))
  (define edges (make-hash))
  (for ([i (in-range n)])
    (define p (vector-ref points i))
    (define nearest
      (~> (range n) (sort < #:key (lambda (j) (squared-distance p (vector-ref points j)))) (take k)))
    (for ([j (in-list nearest)])
      (hash-update! edges (cons i j) (lambda (w) (+ w 0.5)) 0.0)
      (hash-update! edges (cons j i) (lambda (w) (+ w 0.5)) 0.0)))
  (for ([i (in-range n)])
    (hash-remove! edges (cons i i)))
  (edges->graph edges n))

(define (cliques sizes)
  (define edges (make-hash))
  (define starts
    (for/fold ([at 0]
               [starts '()]
               #:result (reverse starts))
              ([s (in-list sizes)])
      (values (+ at s) (cons at starts))))
  (for* ([(s o) (in-parallel (in-list sizes) (in-list starts))]
         [i (in-range s)]
         [j (in-range (add1 i) s)])
    (set-edge! edges (+ o i) (+ o j) 1.0))
  (values (edges->graph edges (apply + sizes)) (truth-of sizes)))

(define (graph-dense g)
  (define n (graph-n g))
  (define m
    (for/vector #:length n
                ([_ (in-range n)])
      (make-vector n 0.0)))
  (for ([i (in-list (graph-rows g))]
        [j (in-list (graph-cols g))]
        [w (in-list (graph-vals g))])
    (vector-set! (vector-ref m i) j (+ w (vector-ref (vector-ref m i) j))))
  (for/list ([row (in-vector m)])
    (vector->list row)))

(define (graph->vectors g)
  (list (list->device-vector (graph-rows g) #:dtype 'int32)
        (list->device-vector (graph-cols g) #:dtype 'int32)
        (list->device-vector (graph-vals g) #:dtype 'float32)))

(define labels
  (make-keyword-procedure (lambda (kws kw-args X)
                            (~> (keyword-apply spectral-clustering-fit kws kw-args (list X))
                                spectral-clustering-result-labels
                                device-vector->list))))

(define graph-labels
  (make-keyword-procedure (lambda (kws kw-args g)
                            (~> (keyword-apply spectral-clustering-fit-graph
                                               kws
                                               kw-args
                                               (append (graph->vectors g) (list (graph-n g))))
                                spectral-clustering-result-labels
                                device-vector->list))))

(define (permute-rows rows perm)
  (define v (list->vector rows))
  (map (lambda (i) (vector-ref v i)) perm))

(define (permute-graph g perm)
  (define new-of (make-vector (length perm) 0))
  (for ([old (in-list perm)]
        [new (in-naturals)])
    (vector-set! new-of old new))
  (define edges (make-hash))
  (for ([i (in-list (graph-rows g))]
        [j (in-list (graph-cols g))]
        [w (in-list (graph-vals g))])
    (hash-set! edges (cons (vector-ref new-of i) (vector-ref new-of j)) w))
  (edges->graph edges (graph-n g)))

(define (graph-shuffle g seed)
  (define shuffled
    (seeded
     seed
     (lambda ()
       (~> (entries g) (map (lambda (e) (cons (random) e)) _) (sort < #:key car) (map cdr _)))))
  (entries->graph shuffled (graph-n g)))

(define (graph-map-vals g f)
  (struct-copy graph g [vals (map f (graph-vals g))]))

(define (entries g)
  (map list (graph-rows g) (graph-cols g) (graph-vals g)))

(define (entries->graph es n)
  (graph (map car es) (map cadr es) (map caddr es) n))

(define (graph-upper g)
  (entries->graph (filter (match-lambda
                            [(list i j _) (< i j)])
                          (entries g))
                  (graph-n g)))

(define (graph-scale-upper g scale)
  (entries->graph (map (match-lambda
                         [(list i j w)
                          (list i
                                j
                                (if (< i j)
                                    (* w scale)
                                    w))])
                       (entries g))
                  (graph-n g)))

(define (repeat-each rows copies)
  (for*/list ([r (in-list rows)]
              [_ (in-range copies)])
    r))
