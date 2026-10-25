#lang racket/base

(require (only-in racket/list last)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in "private/pca-data.rkt" fixture-name fixtures)
         (only-in "private/pca-twin-data.rkt"
                  fit-ours
                  l14-problems
                  ours-noise
                  twin-case
                  twin-ks
                  twin-results)
         (only-in "private/python-env.rkt" test-twin))

(define (cases-of f)
  (match-define (list _ middle _) (twin-ks f))
  (for*/list ([dt (in-list '(float32 float64))]
              [k (in-list (twin-ks f))]
              [whiten (in-list (if (= k middle) '(#f #t) '(#f)))])
    (list f dt k whiten)))

(define (check-against-sklearn f)
  (define cases (cases-of f))
  (define results
    (twin-results (for/list ([c (in-list cases)])
                    (match-define (list f dt k whiten) c)
                    (twin-case "sklearn" f dt k #:whiten whiten))))
  (for* ([(c theirs) (in-parallel (in-list cases) (in-list results))]
         [solver (in-list '(full jacobi))])
    (match-define (list f dt k whiten) c)
    (test-case (format "~a ~a k=~a ~a~a" (fixture-name f) dt k solver (if whiten " whitened" ""))
      (define mine (fit-ours f dt k #:whiten whiten #:solver solver))
      (check-equal? (l14-problems mine theirs theirs dt solver #:whiten whiten) '())
      (define noise (ours-noise mine))
      (cond
        [(= k (last (twin-ks f))) (check-equal? noise 0.0 "L12: noise at k = m")]
        [else (check-true (>= noise 0.0) "L12: noise is never negative")]))))

(for ([f (in-list fixtures)])
  (test-twin (format "L12, L14: ~a is sklearn's within the analytic bounds" (fixture-name f))
    (check-against-sklearn f)))

