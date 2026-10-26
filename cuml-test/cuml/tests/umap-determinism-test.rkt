#lang racket/base

(require (only-in racket/list first remove-duplicates take)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/program.rkt" run-program))

(define (program configs repeats)
  `(module m racket/base
     (require (only-in cuml/manifold umap-fit umap-result-embedding umap-transform)
              (only-in cuml/tests/private/umap-data far-blobs fixture-X fixture-y held-out-fit)
              (only-in file/sha1 sha1)
              (only-in racket/match match-define)
              (only-in raft device-matrix->list*))
     (define fixtures (hasheq 'far-blobs far-blobs 'held-out-fit held-out-fit))
     (define (digest m)
       (sha1 (open-input-string (format "~s" (device-matrix->list* m)))))
     (define (run config)
       (match-define (list name metric init k supervised) config)
       (define f (hash-ref fixtures name))
       (define X (fixture-X f))
       (define result
         (umap-fit X #:y (and supervised (fixture-y f)) #:metric metric #:init init #:n-components k))
       (list (digest (umap-result-embedding result)) (digest (umap-transform result X))))
     (write (for/list ([config (in-list ',configs)])
              (for/list ([_ (in-range ,repeats)])
                (run config))))))

(define (digests configs repeats)
  (match-define (list status out err) (run-program (program configs repeats)))
  (check-equal? status 0 err)
  (read (open-input-string out)))

(define configs
  (append (for*/list ([metric (in-list '(euclidean sqeuclidean manhattan chebyshev cosine))]
                      [init (in-list '(spectral random))]
                      [k (in-list '(2 3 16))]
                      [supervised (in-list '(#f #t))])
            (list 'far-blobs metric init k supervised))
          (for*/list ([init (in-list '(spectral random))]
                      [supervised (in-list '(#f #t))])
            (list 'held-out-fit 'euclidean init 600 supervised))))

(test-gpu "L5: brute-force fits and transforms repeat bitwise in one process and across three"
  (define here (digests configs 3))
  (for ([config (in-list configs)]
        [runs (in-list here)])
    (test-case (format "~a" config)
      (check-equal? (remove-duplicates runs) (take runs 1))))
  (define firsts (map first here))
  (for ([_ (in-range 2)])
    (check-equal? (map first (digests configs 1)) firsts)))
