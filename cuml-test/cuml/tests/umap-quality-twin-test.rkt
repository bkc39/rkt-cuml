#lang racket/base

(require (only-in cuml/manifold
                  umap-fit
                  umap-result-a
                  umap-result-b
                  umap-result-embedding
                  umap-result-params
                  umap-transform)
         (only-in cuml/metrics trustworthiness)
         (only-in racket/list range)
         (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-true test-case)
         (only-in "private/python-env.rkt" run-twin test-twin)
         (only-in "private/umap-data.rkt"
                  blobs
                  fixture-X
                  fixture-name
                  fixture-rows
                  held-out-fit
                  held-out-new
                  overlap
                  twin-case))

(define-runtime-path twin "python/umap_twin.py")

(define (median xs)
  (define sorted (sort xs <))
  (define n (length sorted))
  (/ (+ (list-ref sorted (quotient (sub1 n) 2)) (list-ref sorted (quotient n 2))) 2))

(define (check-within name ours spread)
  (define floor (apply min spread))
  (define delta (- (apply max spread) floor))
  (define centre (median spread))
  (printf "~a: ours ~a, Python min ~a median ~a delta ~a\n" name ours floor centre delta)
  (check-true (>= ours (- floor delta)) (format "~a: ~a below ~a - ~a" name ours floor delta))
  (check-true (<= (abs (- ours centre)) delta) (format "~a: ~a off the median ~a" name ours centre)))

(define (quality-case f result #:transform [Y #f])
  (hash-set* (twin-case f
                        (umap-result-params result)
                        (umap-result-a result)
                        (umap-result-b result)
                        #:transform Y
                        #:op "quality")
             'seeds
             (range 20)))

(test-twin "L7: trustworthiness of fits and transforms is within Python cuML's seed spread"
  (for* ([init (in-list '(spectral random))])
    (define fits
      (for/list ([f (in-list (list blobs overlap))])
        (cons f (umap-fit (fixture-X f) #:init init))))
    (define held (umap-fit (fixture-X held-out-fit) #:init init))
    (define found
      (hash-ref (run-twin twin
                          (hasheq 'cases (append (for/list ([entry (in-list fits)])
                                                   (quality-case (car entry) (cdr entry)))
                                                 (list (quality-case held-out-fit
                                                                     held
                                                                     #:transform
                                                                     (fixture-rows held-out-new))))))
                'results))
    (for ([entry (in-list fits)]
          [theirs (in-list found)])
      (match-define (cons f result) entry)
      (test-case (format "~a fit ~a" init (fixture-name f))
        (check-within (format "~a fit" init)
                      (trustworthiness (fixture-X f) (umap-result-embedding result) #:n-neighbors 10)
                      (hash-ref theirs 'trustworthiness))))
    (check-within (format "~a transform" init)
                  (trustworthiness (fixture-X held-out-new)
                                   (umap-transform held (fixture-X held-out-new))
                                   #:n-neighbors 10)
                  (hash-ref (list-ref found 2) 'trustworthiness))))
