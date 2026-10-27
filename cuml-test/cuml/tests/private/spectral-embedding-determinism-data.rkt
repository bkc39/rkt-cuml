#lang racket/base

(require (only-in "umap-determinism-data.rkt" check-repeatable))

(provide check-se-repeatable
         dense-configs
         graph-configs)

(define ((se-program given) configs repeats)
  `(module m racket/base
     (require (only-in cuml/manifold spectral-embedding-fit spectral-embedding-result-embedding)
              (only-in cuml/tests/private/estimator-data apply-options)
              (only-in cuml/tests/private/spectral-data three-blocks)
              (only-in cuml/tests/private/spectral-embedding-data
                       embedded-graph
                       rows-matrix
                       stochastic-graph)
              (only-in cuml/tests/private/umap-data blobs fixture-rows)
              (only-in cuml/tests/private/umap-determinism-data digest)
              (only-in racket/match match-define))
     (define (blocks)
       (define-values (g _truth) (three-blocks))
       g)
     (define fixtures
       (hasheq 'blobs (lambda () (fixture-rows blobs))
               'given (lambda () ',given)
               'sbm stochastic-graph
               'three-blocks blocks))
     (define (run config)
       (match-define (list kind name options) config)
       (define data ((hash-ref fixtures name)))
       (define result
         (if (eq? kind 'dense)
             (apply-options spectral-embedding-fit options (rows-matrix data))
             (embedded-graph data options)))
       (digest (spectral-embedding-result-embedding result)))
     (write (for/list ([config (in-list ',configs)])
              (for/list ([_ (in-range ,repeats)])
                (run config))))))

(define (settings)
  (for*/list ([drop (in-list '(#t #f))]
              [norm (in-list '(#t #f))])
    (hasheq 'drop-first drop 'norm-laplacian norm)))

(define (dense-configs name)
  (for*/list ([n-neighbors (in-list '(10 256))]
              [options (in-list (settings))])
    (list 'dense name (hash-set options 'n-neighbors n-neighbors))))

(define (graph-configs name)
  (for/list ([options (in-list (settings))])
    (list 'graph name options)))

(define (check-se-repeatable configs #:given [given '()])
  (check-repeatable configs #:program (se-program given)))
