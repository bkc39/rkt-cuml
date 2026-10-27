#lang racket/base

(require (only-in cuml/manifold
                  tsne-fit
                  tsne-params
                  tsne-result-embedding
                  tsne-result-kl-divergence
                  tsne-result-n-iter
                  tsne-result-params
                  tsne-result?)
         (only-in racket/flonum flsingle)
         (only-in rackunit check-equal? check-false check-pred check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/scaler-data.rkt" bits-equal? matrix-shape)
         (only-in "private/tsne-data.rkt" embedded)
         (only-in "private/umap-data.rkt" far-blobs finite-rows? fixture-X fixture-rows overlap tiny))

(define (single? x)
  (and (flonum? x) (eqv? x (flsingle x))))

(define (check-statistics result max-iter)
  (define n-iter (tsne-result-n-iter result))
  (define kl (tsne-result-kl-divergence result))
  (check-true (and (exact-integer? n-iter) (<= 1 n-iter max-iter)) (format "n-iter ~a" n-iter))
  (check-equal? (not kl) (< n-iter max-iter) (format "kl ~a at n-iter ~a" kl n-iter))
  (check-true (or (not kl) (and (single? kl) (>= kl 0.0))) (format "kl ~a" kl)))

(test-gpu "L1: the embedding is a fresh n×2 float32 matrix, with the run's statistics and params"
  (for* ([method (in-list '(fft exact))]
         [init (in-list '(random pca))])
    (test-case (format "~a ~a" method init)
      (define X (fixture-X far-blobs))
      (define result (tsne-fit X #:method method #:init init #:max-iter 500))
      (check-true (tsne-result? result))
      (check-equal? (matrix-shape (tsne-result-embedding result))
                    (list #t 'float32 'row-major (list 300 2)))
      (check-false (eq? (tsne-result-embedding result) X))
      (check-true (finite-rows? (embedded result)))
      (check-statistics result 500)
      (check-equal? (tsne-result-params result)
                    (tsne-params #:method method #:init init #:max-iter 500)))))

(test-gpu "L1: an FFT run is never cut short; its KL is cuML's float32 value"
  (define result (tsne-fit (fixture-X far-blobs) #:max-iter 300))
  (check-equal? (tsne-result-n-iter result) 300)
  (check-pred single? (tsne-result-kl-divergence result)))

(define (settled-fit max-iter)
  (tsne-fit (fixture-X tiny)
            #:method 'exact
            #:perplexity 5.0
            #:learning-rate 1.0
            #:early-exaggeration 1.0
            #:max-iter max-iter))

(define settled-stop 261)

(test-gpu "L1: exact stops on a small gradient at a checked iteration, without a KL"
  (define result (settled-fit 1000))
  (define n-iter (tsne-result-n-iter result))
  (check-true (< 251 n-iter 1000) (format "n-iter ~a" n-iter))
  (check-equal? (modulo (sub1 n-iter) 10) 0 (format "n-iter ~a" n-iter))
  (check-false (tsne-result-kl-divergence result)))

(test-gpu "L1: a stop on the last iteration counts as a full run, with its KL"
  (define result (settled-fit settled-stop))
  (check-equal? (tsne-result-n-iter result) settled-stop)
  (check-pred single? (tsne-result-kl-divergence result)))

(test-gpu "L1: params keep a derived n-neighbors and a drawn seed as #f"
  (define params (tsne-result-params (tsne-fit (fixture-X far-blobs) #:random-state #f)))
  (check-equal? (hash-ref params 'n-neighbors) #f)
  (check-equal? (hash-ref params 'random-state) #f))

(test-gpu "L2: fit leaves X bitwise unchanged under both inits"
  (for* ([method (in-list '(fft exact))]
         [init (in-list '(random pca))])
    (define X (fixture-X overlap))
    (tsne-fit X #:method method #:init init #:max-iter 300)
    (check-true (bits-equal? (device-matrix->list* X) (fixture-rows overlap))
                (format "~a ~a" method init))))
