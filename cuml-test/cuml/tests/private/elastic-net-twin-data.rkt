#lang racket/base

(require (only-in racket/match match-define struct*)
         racket/runtime-path
         (only-in rackunit check-equal? test-case)
         (only-in "elastic-net-data.rkt"
                  correlated-case
                  distance
                  dyadic-weight-case
                  en-case-fit
                  law-alpha
                  mirrored-weight-case
                  selections)
         (only-in "elastic-net-oracle.rkt" en-solution oracle-elastic-net problem-of s2-bound)
         (only-in "linear-data.rkt" check-law lcase-name lcase-w lcase-X lcase-y unit-roundoff)
         (only-in "linear-twin-data.rkt" case-request theirs twin-results))

(provide check-raw-parity
         check-sklearn-parity)

(define-runtime-path twin "../python/elastic_net_twin.py")

(define (request op c dt alpha r fi s #:max-iter [max-iter 1000] #:tol [tol 1e-3])
  (case-request op
                c
                dt
                fi
                (hasheq 'alpha (real->double-flonum alpha)
                        'l1_ratio (real->double-flonum r)
                        'max_iter max-iter
                        'tol tol
                        'selection (symbol->string s))))

(define (alpha-of c fi r)
  (law-alpha (problem-of (lcase-X c) (lcase-y c) (lcase-w c) fi) 0.01 r))

(define raw-sources
  (list (correlated-case 200 7 0.9 41) (correlated-case 20 2 0 42) (correlated-case 2000 7 0.5 43)))

(define (raw-rows)
  (append (for*/list ([c (in-list raw-sources)]
                      [fi (in-list '(#t #f))]
                      [r (in-list '(1 0.5 0))]
                      [s (in-list selections)])
            (list c fi r s))
          (for*/list ([seed (in-list '(44 45))]
                      [r (in-list '(1 0.5))]
                      [s (in-list selections)]
                      [row (in-list (list (list (dyadic-weight-case seed) #f)
                                          (list (mirrored-weight-case seed) #t)))])
            (match-define (list c fi) row)
            (list c fi r s))))

(define (check-raw-parity dt)
  (define rows (raw-rows))
  (define fits
    (twin-results twin
                  (for/list ([row (in-list rows)])
                    (match-define (list c fi r s) row)
                    (request "cuml" c dt (alpha-of c fi r) r fi s))))
  (for ([row (in-list rows)]
        [t (in-list fits)])
    (match-define (list c fi r s) row)
    (test-case (format "~a ~a ~a l1-ratio ~a ~a" (lcase-name c) dt fi r s)
      (match-define (list coef b k)
        (en-case-fit c dt #:alpha (alpha-of c fi r) #:l1-ratio r #:fit-intercept fi #:selection s))
      (match-define (list coef* b*) (theirs dt t))
      (check-equal? coef coef*)
      (check-equal? k (hash-ref t 'n_iter))
      (unless (lcase-w c)
        (check-equal? b b*)))))

(define sklearn-sources
  (list (correlated-case 200 7 0.9 46 #:weighted? #t) (correlated-case 200 4 0 47)))

(define (check-sklearn-parity dt)
  (define rows
    (for*/list ([c (in-list sklearn-sources)]
                [fi (in-list '(#t #f))]
                [r (in-list '(1 0.5 0.05))])
      (list c fi r)))
  (define fits
    (twin-results twin
                  (for/list ([row (in-list rows)])
                    (match-define (list c fi r) row)
                    (request "sklearn" c dt (alpha-of c fi r) r fi 'cyclic))))
  (define ran
    (for/sum
     ([row (in-list rows)] [t (in-list fits)])
     (match-define (list c fi r) row)
     (define p (problem-of (lcase-X c) (lcase-y c) (lcase-w c) fi))
     (define alpha (alpha-of c fi r))
     (define o (oracle-elastic-net p alpha r))
     (match-define (struct* en-solution ([coef c*] [mu mu])) o)
     (match-define (list coef _ k) (en-case-fit c dt #:alpha alpha #:l1-ratio r #:fit-intercept fi))
     (define theirs-coef (map inexact->exact (hash-ref t 'coef)))
     (cond
       [(and (< k 1000) (> mu 0))
        (test-case (format "~a ~a ~a l1-ratio ~a" (lcase-name c) dt fi r)
          (check-law "S4"
                     (distance coef theirs-coef)
                     (+ (s2-bound o coef alpha r 1e-3 (unit-roundoff dt)) (distance theirs-coef c*))))
        1]
       [else 0])))
  (test-case (format "S4 ran on every ~a fit" dt)
    (check-equal? ran (length rows))))
