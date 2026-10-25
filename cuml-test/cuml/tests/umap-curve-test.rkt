#lang racket/base

(require (only-in cuml/manifold/private/curve curve-iterations curve-parameters)
         (only-in racket/flonum fl* flexpt)
         (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-equal? check-true test-case)
         (only-in "private/python-env.rkt" run-twin test-twin))

(define-runtime-path twin "python/umap_twin.py")

(define pairs
  (for*/list ([s (in-list '(0.5 1.0 1.5 2.0 3.0 5.0))]
              [m (in-list '(0.0 0.001 0.01 0.05 0.1 0.25 0.5 0.99))]
              #:when (<= m s))
    (cons s m)))

(define (relative ours theirs)
  (/ (abs (- ours theirs)) (abs theirs)))

(define curve-fit-bound (sqrt 1.49012e-8))
(define tight-bound 1e-7)

(test-case "F3 covers 47 pairs"
  (check-equal? (length pairs) 47))

(test-twin "L4a, L4b: a and b are within the bounds of scipy's curve_fit and a precise solver"
  (define found
    (hash-ref (run-twin twin
                        (hasheq 'cases (for/list ([p (in-list pairs)])
                                         (hasheq 'op "ab" 'spread (car p) 'min_dist (cdr p)))))
              'results))
  (for ([p (in-list pairs)]
        [theirs (in-list found)])
    (match-define (cons s m) p)
    (define-values (a b) (curve-parameters m s))
    (test-case (format "spread ~a min-dist ~a" s m)
      (check-true (<= (relative a (hash-ref theirs 'a)) curve-fit-bound) "curve_fit a")
      (check-true (<= (relative b (hash-ref theirs 'b)) curve-fit-bound) "curve_fit b")
      (check-true (<= (relative a (hash-ref theirs 'tight_a)) tight-bound) "tight a")
      (check-true (<= (relative b (hash-ref theirs 'tight_b)) tight-bound) "tight b"))))

(test-case "L4c: at most 50 iterations over min-dist/spread in [0, 1]"
  (check-true (for/and ([i (in-range 1001)])
                (<= (curve-iterations (/ i 1000.0) 1.0) 50))))

(test-case "L4d: the fit is in normalised coordinates"
  (for* ([s (in-list '(1e-6 1e-3 1e3 1e6))]
         [m (in-list (list 0.0 (* 0.1 s) (* 0.5 s)))])
    (define-values (a b) (curve-parameters m s))
    (define-values (a1 b1) (curve-parameters (/ m s) 1.0))
    (test-case (format "spread ~a min-dist ~a" s m)
      (check-equal? b b1)
      (check-equal? a (fl* a1 (flexpt s (* -2.0 b1)))))))

(test-case "L4e: a tiny spread fits as spread 1 does, where scipy's b is -70.5"
  (define-values (_a b) (curve-parameters 1e-4 1e-3))
  (define-values (_a1 b1) (curve-parameters 0.1 1.0))
  (check-equal? b b1)
  (check-true (< (abs (- b 0.89506)) 1e-5)))

(test-twin "L4e: scipy's departure at spread 1e-3, min-dist 1e-4 is pinned"
  (match-define (list theirs)
    (hash-ref (run-twin twin (hasheq 'cases (list (hasheq 'op "ab" 'spread 1e-3 'min_dist 1e-4))))
              'results))
  (check-true (< (abs (+ (hash-ref theirs 'b) 70.497)) 1e-3)))
