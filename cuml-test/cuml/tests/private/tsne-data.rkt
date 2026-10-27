#lang racket/base

(require (only-in cuml/core exn:fail:cuml?)
         (only-in cuml/manifold
                  tsne-fit
                  tsne-result
                  tsne-result-embedding
                  tsne-result-kl-divergence
                  tsne-result-n-iter
                  tsne-result?)
         (only-in cuml/metrics trustworthiness)
         (only-in racket/match match-define)
         (only-in racket/math exact-ceiling)
         racket/runtime-path
         (only-in rackunit check-equal? check-regexp-match check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in "blob-data.rkt" centred-blobs uniform-source)
         (only-in "estimator-data.rkt" apply-options)
         (only-in "python-env.rkt" run-cases)
         (only-in "scaler-data.rkt" bits-equal?)
         (only-in "umap-data.rkt" far-blobs finite-rows? fixture fixture-name fixture-rows fixture-X))

(provide check-quality
         check-sklearn
         check-small-sweep
         check-tsne-parity
         derived-neighbours
         diverged
         embedded
         tsne-case
         twin
         wide-blobs)

(define-runtime-path twin "../python/tsne_twin.py")

(define wide-blobs
  (let-values ([(rows labels) (centred-blobs (let ([next (uniform-source 54)])
                                               (for/list ([_ (in-range 5)])
                                                 (for/list ([_ (in-range 50)])
                                                   (+ -10.0 (* 20.0 (next))))))
                                             2400
                                             55)])
    (fixture "wide-blobs" rows labels)))

(define sweep-sizes '(2 3 4 5 6 7 8 10 12 16 20 30 40))

(define (small-sweep)
  (for*/list ([n (in-list sweep-sizes)]
              [seed (in-range 8)])
    (define next (uniform-source (+ 1000 (* 8 n) seed)))
    (fixture (format "sweep n=~a seed=~a" n seed)
             (for/list ([_ (in-range n)])
               (for/list ([_ (in-range 4)])
                 (next)))
             (for/list ([_ (in-range n)])
               0))))

(define (diverged caller)
  (pregexp
   (format
    "^~a: embedding: row [0-9]+, column [0-9]+: the optimisation diverged to (nan|inf|-inf); lower learning-rate$"
    caller)))

(define (check-small-sweep method init)
  (for ([f (in-list (small-sweep))])
    (define n (length (fixture-rows f)))
    (define outcome
      (with-handlers ([exn:fail:cuml? values])
        (tsne-fit (fixture-X f) #:method method #:init init #:perplexity (min 30.0 n))))
    (test-case (format "~a ~a ~a" method init (fixture-name f))
      (cond
        [(tsne-result? outcome) (check-true (finite-rows? (embedded outcome)))]
        [else (check-regexp-match (diverged 'tsne-fit) (exn-message outcome))])))
  (check-true (finite-rows? (embedded (tsne-fit (fixture-X far-blobs) #:method method #:init init)))
              "a valid call after the sweep"))

(define (derived-neighbours perplexity)
  (min 1023 (max 2 (exact-ceiling (* 3 perplexity)))))

(define (embedded result)
  (device-matrix->list* (tsne-result-embedding result)))

(define (tsne-case f result #:op [op "tsne"] #:neighbours [k #f])
  (match-define (tsne-result _ _ _ params) result)
  (define (given key)
    (hash-ref params key))
  (hasheq 'op op
          'X (fixture-rows f)
          'perplexity (given 'perplexity)
          'early_exaggeration (given 'early-exaggeration)
          'learning_rate (given 'learning-rate)
          'max_iter (given 'max-iter)
          'n_neighbors (or k (given 'n-neighbors) (derived-neighbours (given 'perplexity)))
          'method (symbol->string (given 'method))
          'metric (symbol->string (given 'metric))
          'init (symbol->string (given 'init))
          'seed (given 'random-state)))

(define (check-tsne-parity f configs #:neighbours [k #f])
  (define results
    (for/list ([options (in-list configs)])
      (apply-options tsne-fit options (fixture-X f))))
  (define theirs
    (run-cases twin
               (for/list ([result (in-list results)])
                 (tsne-case f result #:neighbours k))))
  (for ([options (in-list configs)]
        [result (in-list results)]
        [found (in-list theirs)])
    (test-case (format "~a ~a" (fixture-name f) options)
      (check-true (bits-equal? (embedded result) (hash-ref found 'embedding)) "embedding")
      (check-equal? (tsne-result-kl-divergence result) (hash-ref found 'kl) "kl")
      (check-equal? (tsne-result-n-iter result) (hash-ref found 'n_iter) "n-iter"))))

(define quality-floors
  (hasheq 'fft (hash "blobs" 0.965 "overlap" 0.826 "digits" 0.9858)
          'exact (hash "blobs" 0.969 "overlap" 0.845 "digits" 0.9888)))

(define sklearn-bounds (hash "blobs" 0.008 "overlap" 0.07 "digits" 0.004))

(define (trust-of f result)
  (trustworthiness (fixture-X f) (tsne-result-embedding result) #:n-neighbors 10))

(define (check-quality method fixtures)
  (for ([f (in-list fixtures)])
    (define name (format "~a ~a" method (fixture-name f)))
    (define t (trust-of f (tsne-fit (fixture-X f) #:method method)))
    (define least (hash-ref (hash-ref quality-floors method) (fixture-name f)))
    (test-case name
      (check-true (>= t least) (format "~a: ~a below ~a" name t least)))))

(define (check-sklearn fixtures)
  (define results
    (for/list ([f (in-list fixtures)])
      (tsne-fit (fixture-X f) #:method 'exact)))
  (define references
    (run-cases twin
               (for/list ([f (in-list fixtures)]
                          [result (in-list results)])
                 (tsne-case f result #:op "sklearn"))))
  (for ([f (in-list fixtures)]
        [result (in-list results)]
        [reference (in-list references)])
    (define name (format "exact ~a against sklearn" (fixture-name f)))
    (define t (trust-of f result))
    (define theirs (hash-ref reference 'trustworthiness))
    (define bound (hash-ref sklearn-bounds (fixture-name f)))
    (test-case name
      (check-true (<= (abs (- t theirs)) bound)
                  (format "~a: ~a against ~a, past ~a" name t theirs bound)))))
