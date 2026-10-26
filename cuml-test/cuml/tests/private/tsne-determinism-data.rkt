#lang racket/base

(require (only-in "umap-determinism-data.rkt" check-repeatable))

(provide check-tsne-repeatable)

(define (tsne-program configs repeats)
  `(module m racket/base
     (require (only-in cuml/manifold
                       tsne-fit
                       tsne-result-embedding
                       tsne-result-kl-divergence
                       tsne-result-n-iter)
              (only-in cuml/tests/private/umap-data digits far-blobs fixture-X)
              (only-in cuml/tests/private/umap-determinism-data digest)
              (only-in racket/match match-define))
     (define fixtures (hasheq 'far-blobs (lambda () far-blobs) 'digits digits))
     (define (run X metric init)
       (define result (tsne-fit X #:metric metric #:init init))
       (list (digest (tsne-result-embedding result))
             (tsne-result-kl-divergence result)
             (tsne-result-n-iter result)))
     (write (for/list ([config (in-list ',configs)])
              (match-define (list name metric init) config)
              (define X (fixture-X ((hash-ref fixtures name))))
              (for/list ([_ (in-range ,repeats)])
                (run X metric init))))))

(define (check-tsne-repeatable name metrics)
  (check-repeatable (for*/list ([metric (in-list metrics)]
                                [init (in-list '(random pca))])
                      (list name metric init))
                    #:program tsne-program))
