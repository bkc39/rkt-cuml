#lang racket/base

(require (only-in cuml/manifold
                  spectral-embedding-fit
                  spectral-embedding-fit-graph
                  spectral-embedding-result-embedding
                  spectral-embedding-result-n-neighbors
                  spectral-embedding-result-params)
         (only-in racket/list append* argmax second)
         (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-true test-case)
         (only-in raft device-matrix->list* list*->device-matrix)
         (only-in "dbscan-data.rkt" seeded)
         (only-in "estimator-data.rkt" apply-options)
         (only-in "python-env.rkt" run-cases)
         (only-in "scaler-data.rkt" bits-equal?)
         (only-in "spectral-data.rkt" graph graph->vectors graph-cols graph-n graph-rows graph-vals))

(provide check-twins
         column-gaps
         columns
         cosine-gaps
         embedded
         embedded-graph
         graph-case
         graph-checks
         rows-matrix
         sbm
         se-case
         se-checks
         sign-ruled?
         twin)

(define-runtime-path twin "../python/spectral_embedding_twin.py")

(define (sbm sizes p-in p-out seed #:weighted? [weighted? #f])
  (define truth
    (append* (for/list ([s (in-list sizes)]
                        [g (in-naturals)])
               (for/list ([_ (in-range s)])
                 g))))
  (define n (length truth))
  (define groups (list->vector truth))
  (define edges
    (seeded seed
            (lambda ()
              (for*/list ([i (in-range n)]
                          [j (in-range (add1 i) n)]
                          #:when (< (random)
                                    (if (= (vector-ref groups i) (vector-ref groups j)) p-in p-out)))
                (list i
                      j
                      (if weighted?
                          (+ 0.5 (* 1.5 (random)))
                          1.0))))))
  (define both
    (sort (append* (for/list ([e (in-list edges)])
                     (match-define (list i j w) e)
                     (list (list i j w) (list j i w))))
          (lambda (a b) (or (< (car a) (car b)) (and (= (car a) (car b)) (< (cadr a) (cadr b)))))))
  (values (graph (map car both) (map cadr both) (map caddr both) n) truth))

(define (rows-matrix rows)
  (list*->device-matrix rows #:dtype 'float32))

(define (embedded result)
  (device-matrix->list* (spectral-embedding-result-embedding result)))

(define (embedded-graph g [options (hasheq)])
  (apply apply-options
         spectral-embedding-fit-graph
         options
         (append (graph->vectors g) (list (graph-n g)))))

(define (columns rows)
  (apply map list rows))

(define (sign-ruled? rows)
  (for/and ([column (in-list (columns rows))])
    (> (argmax abs column) 0.0)))

(define (largest column)
  (for/fold ([best 0.0]) ([x (in-list column)])
    (max best (abs x))))

(define (column-gaps ours theirs)
  (for/list ([e (in-list (columns ours))]
             [s (in-list (columns theirs))])
    (/ (for/fold ([gap 0.0])
                 ([a (in-list e)]
                  [b (in-list s)])
         (max gap (abs (- a b))))
       (largest s))))

(define (dot a b)
  (for/sum ([x (in-list a)] [y (in-list b)]) (* x y)))

(define (cosine-gaps ours theirs)
  (for/list ([e (in-list (columns ours))]
             [s (in-list (columns theirs))])
    (- 1.0 (/ (abs (dot e s)) (sqrt (* (dot e e) (dot s s)))))))

(define (se-case X result #:op [op "embed"])
  (define params (spectral-embedding-result-params result))
  (hasheq 'op op
          'X X
          'k (hash-ref params 'n-components)
          'n_neighbors (spectral-embedding-result-n-neighbors result)
          'seed (hash-ref params 'random-state)
          'norm (hash-ref params 'norm-laplacian)
          'drop (hash-ref params 'drop-first)))

(define (graph-case g options #:op [op "graph"])
  (hasheq 'op op
          'rows (graph-rows g)
          'cols (graph-cols g)
          'vals (graph-vals g)
          'n (graph-n g)
          'k (hash-ref options 'n-components 2)
          'seed (hash-ref options 'random-state 0)
          'norm (hash-ref options 'norm-laplacian #t)
          'drop (hash-ref options 'drop-first #t)))

(define ((bitwise ours) theirs)
  (check-true (bits-equal? ours theirs) "embedding"))

(define (se-checks name rows configs)
  (define X (rows-matrix rows))
  (for/list ([options (in-list configs)])
    (define result (apply-options spectral-embedding-fit options X))
    (list (format "~a ~a" name options) (se-case rows result) (bitwise (embedded result)))))

(define (graph-checks name g configs)
  (for/list ([options (in-list configs)])
    (list (format "~a ~a" name options)
          (graph-case g options)
          (bitwise (embedded (embedded-graph g options))))))

(define (check-twins entries)
  (for ([entry (in-list entries)]
        [found (in-list (run-cases twin (map second entries)))])
    (match-define (list label _ check) entry)
    (test-case label
      (check (hash-ref found 'embedding)))))
