#lang racket/base

(require (only-in cuml/metrics trustworthiness)
         (only-in racket/list remove-duplicates)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list* list*->device-matrix)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/metrics-data.rkt" grid-20 permute projection random-permutation))

(define n 200)

(define (fixture)
  (projection n 10 2))

(define (scaled X by)
  (list*->device-matrix (for/list ([row (in-list (device-matrix->list* X))])
                          (for/list ([x (in-list row)])
                            (* x by)))
                        #:dtype 'float32))

(define (permuted X order)
  (list*->device-matrix (permute (device-matrix->list* X) order) #:dtype 'float32))

(test-gpu "TW1: an embedding of X by itself scores exactly 1.0"
  (define-values (X _) (fixture))
  (check-equal? (trustworthiness X X) 1.0)
  (check-equal? (trustworthiness X X #:n-neighbors 30) 1.0))

(test-gpu "TW2: the score lies in [0, 1] for every n-neighbors below half the rows"
  (define-values (X E) (projection 60 10 2))
  (for ([k (in-range 1 30)])
    (define s (trustworthiness X E #:n-neighbors k))
    (check-true (<= 0.0 s 1.0) (format "k ~a: ~a" k s)))
  (define grid (list*->device-matrix grid-20 #:dtype 'float32))
  (define line
    (list*->device-matrix (for/list ([p (in-list grid-20)])
                            (list (car p)))
                          #:dtype 'float32))
  (for ([k (in-list '(1 4 5 8))])
    (check-true (<= 0.0 (trustworthiness grid line #:n-neighbors k) 1.0))))

(test-gpu "TW3: the batch size leaves the score bit-identical"
  (define-values (X E) (fixture))
  (define base (trustworthiness X E))
  (for ([b (in-list (list 1 7 512 (sub1 n) n 4096))])
    (check-equal? (trustworthiness X E #:batch-size b) base (format "batch ~a" b))))

(test-gpu "TW4: ten calls give one value"
  (define-values (X E) (fixture))
  (check-equal? (length (remove-duplicates (for/list ([_ (in-range 10)])
                                             (trustworthiness X E))))
                1))

(test-gpu "TW7: scaling either matrix by a power of two leaves the score bit-identical"
  (define-values (X E) (fixture))
  (define base (trustworthiness X E))
  (for ([by (in-list '(2 1/4 8))])
    (test-case (format "by ~a" by)
      (check-equal? (trustworthiness (scaled X by) E) base)
      (check-equal? (trustworthiness X (scaled E by)) base))))

(test-gpu "TW8: permuting the rows of both alike leaves the score bit-identical"
  (define-values (X E) (fixture))
  (define order (random-permutation n 5))
  (check-equal? (trustworthiness (permuted X order) (permuted E order)) (trustworthiness X E)))
