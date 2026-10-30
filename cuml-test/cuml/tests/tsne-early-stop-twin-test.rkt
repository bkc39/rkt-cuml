#lang racket/base

(require (only-in cuml/manifold
                  tsne-fit
                  tsne-result-embedding
                  tsne-result-kl-divergence
                  tsne-result-n-iter)
         (only-in cuml/metrics trustworthiness)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-pred check-true)
         (only-in "private/python-env.rkt" run-cases test-twin)
         (only-in "private/tsne-data.rkt" tsne-case twin wide-blobs)
         (only-in "private/umap-data.rkt" fixture-X))

(define trust-floor 0.93)

(test-twin "L8: FFT runs every iteration on wide blobs where Python's default stops and collapses"
  (define X (fixture-X wide-blobs))
  (define result (tsne-fit X))
  (check-equal? (tsne-result-n-iter result) 1000)
  (check-pred flonum? (tsne-result-kl-divergence result))
  (define t (trustworthiness X (tsne-result-embedding result) #:n-neighbors 10))
  (check-true (>= t trust-floor) (format "trustworthiness ~a" t))
  (match-define (list theirs)
    (run-cases twin (list (tsne-case wide-blobs result #:op "python-default"))))
  (check-true (< (hash-ref theirs 'n_iter) 250)
              (format "Python's n_iter ~a" (hash-ref theirs 'n_iter)))
  (check-true (< (hash-ref theirs 'std) 1e-3) (format "Python's std ~a" (hash-ref theirs 'std))))
