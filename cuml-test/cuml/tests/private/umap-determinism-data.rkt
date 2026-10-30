#lang racket/base

(require (only-in file/sha1 sha1)
         (only-in racket/list first remove-duplicates take)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in raft device-matrix->list*)
         (only-in "program.rkt" run-program))

(provide check-repeatable
         digest
         narrow-configs
         wide-configs)

(define (digest m)
  (define out (open-output-bytes))
  (for* ([row (in-list (device-matrix->list* m))]
         [x (in-list row)])
    (write-bytes (real->floating-point-bytes x 4) out))
  (sha1 (open-input-bytes (get-output-bytes out))))

(define (umap-program configs repeats)
  `(module m racket/base
     (require (only-in cuml/manifold umap-fit umap-result-embedding umap-transform)
              (only-in cuml/tests/private/umap-data far-blobs fixture-X fixture-y held-out-fit)
              (only-in cuml/tests/private/umap-determinism-data digest)
              (only-in racket/match match-define))
     (define fixtures (hasheq 'far-blobs far-blobs 'held-out-fit held-out-fit))
     (define (run config)
       (match-define (list name metric init k supervised epochs) config)
       (define f (hash-ref fixtures name))
       (define X (fixture-X f))
       (define result
         (umap-fit X
                   #:y (and supervised (fixture-y f))
                   #:metric metric
                   #:init init
                   #:n-components k
                   #:n-epochs epochs))
       (list (digest (umap-result-embedding result)) (digest (umap-transform result X))))
     (write (for/list ([config (in-list ',configs)])
              (for/list ([_ (in-range ,repeats)])
                (run config))))))

(define (digests program configs repeats)
  (match-define (list status out err) (run-program (program configs repeats)))
  (check-equal? status 0 err)
  (read (open-input-string out)))

(define wide-epochs 30)

(define (narrow-configs metrics)
  (for*/list ([metric (in-list metrics)]
              [init (in-list '(spectral random))]
              [k (in-list '(2 3 16))]
              [supervised (in-list '(#f #t))])
    (list 'far-blobs metric init k supervised #f)))

(define (wide-configs init k)
  (for/list ([supervised (in-list '(#f #t))])
    (list 'held-out-fit 'euclidean init k supervised wide-epochs)))

(define (check-repeatable configs #:program [program umap-program])
  (define here (digests program configs 3))
  (for ([config (in-list configs)]
        [runs (in-list here)])
    (test-case (format "~a" config)
      (check-equal? (remove-duplicates runs) (take runs 1))))
  (define firsts (map first here))
  (for ([_ (in-range 2)])
    (check-equal? (map first (digests program configs 1)) firsts)))
