#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/program.rkt" run-program))

(define (program repeats)
  `(module m racket/base
     (require (only-in cuml/manifold umap-fit umap-result-embedding umap-transform)
              (only-in cuml/tests/private/umap-data far-blobs fixture-X fixture-y overlap)
              (only-in file/sha1 sha1)
              (only-in racket/match match-define)
              (only-in raft device-matrix->list*))
     (define (digest m)
       (sha1 (open-input-string (format "~s" (device-matrix->list* m)))))
     (define configs
       (append (for*/list ([metric (in-list '(euclidean sqeuclidean manhattan chebyshev cosine))]
                           [init (in-list '(spectral random))]
                           [k (in-list '(2 3 16))]
                           [supervised (in-list '(#f #t))])
                 (list far-blobs metric init k supervised))
               (for*/list ([init (in-list '(spectral random))]
                           [supervised (in-list '(#f #t))])
                 (list overlap 'euclidean init 600 supervised))))
     (define (run config)
       (match-define (list f metric init k supervised) config)
       (define X (fixture-X f))
       (define result
         (umap-fit X #:y (and supervised (fixture-y f)) #:metric metric #:init init #:n-neighbors k))
       (list (digest (umap-result-embedding result)) (digest (umap-transform result X))))
     (write (for/list ([config (in-list configs)])
              (for/list ([_ (in-range ,repeats)])
                (run config))))))

(define (digests repeats)
  (match-define (list status out err) (run-program (program repeats)))
  (check-equal? status 0 err)
  (read (open-input-string out)))

(test-gpu "L5: brute-force fits and transforms repeat bitwise in one process and across three"
  (define here (digests 3))
  (for ([runs (in-list here)])
    (check-true (andmap (lambda (r) (equal? r (car runs))) runs) (format "~a" runs)))
  (define firsts (map car here))
  (for ([_ (in-range 2)])
    (check-equal? (map car (digests 1)) firsts)))
