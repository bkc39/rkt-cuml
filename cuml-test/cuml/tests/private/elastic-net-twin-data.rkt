#lang racket/base

(require (only-in racket/match match-define struct*)
         racket/runtime-path
         (only-in rackunit check-equal? test-case)
         (only-in "elastic-net-data.rkt"
                  case-alpha
                  coef-distance
                  converged?
                  correlated-case
                  default-max-iter
                  default-tol
                  dyadic-weight-case
                  en-case-fit
                  en-run
                  mirrored-weight-case
                  selections)
         (only-in "elastic-net-oracle.rkt" en-solution oracle-elastic-net problem-of s2-bound)
         (only-in "linear-data.rkt" check-law lcase-name lcase-w lcase-X lcase-y unit-roundoff)
         (only-in "linear-twin-data.rkt" case-request theirs twin-results))

(provide check-raw-parity
         check-sklearn-parity)

(define-runtime-path twin "../python/elastic_net_twin.py")

(struct twin-row (source fit-intercept l1-ratio selection))

(define twin-alpha-ratio 0.01)

(define (row-alpha row)
  (match-define (struct* twin-row ([source c] [fit-intercept fi] [l1-ratio r])) row)
  (case-alpha c twin-alpha-ratio r #:fit-intercept fi))

(define (request op row dt)
  (match-define (struct* twin-row ([source c] [fit-intercept fi] [l1-ratio r] [selection s])) row)
  (case-request op
                c
                dt
                fi
                (hasheq 'alpha (real->double-flonum (row-alpha row))
                        'l1_ratio (real->double-flonum r)
                        'max_iter default-max-iter
                        'tol default-tol
                        'selection (symbol->string s))))

(define raw-sources
  (list (correlated-case 200 7 0.9 41) (correlated-case 20 2 0 42) (correlated-case 2000 7 0.5 43)))

(define (raw-rows)
  (append (for*/list ([c (in-list raw-sources)]
                      [fi (in-list '(#t #f))]
                      [r (in-list '(1 0.5 0))]
                      [s (in-list selections)])
            (twin-row c fi r s))
          (for*/list ([seed (in-list '(44 45))]
                      [r (in-list '(1 0.5))]
                      [s (in-list selections)]
                      [row (in-list (list (twin-row (dyadic-weight-case seed) #f r s)
                                          (twin-row (mirrored-weight-case seed) #t r s)))])
            row)))

(define (check-raw-parity dt)
  (define rows (raw-rows))
  (define fits
    (twin-results twin
                  (for/list ([row (in-list rows)])
                    (request "cuml" row dt))))
  (for ([row (in-list rows)]
        [t (in-list fits)])
    (match-define (struct* twin-row ([source c] [fit-intercept fi] [l1-ratio r] [selection s])) row)
    (test-case (format "~a ~a ~a l1-ratio ~a ~a" (lcase-name c) dt fi r s)
      (match-define (struct* en-run ([coef coef] [intercept b] [n-iter k]))
        (en-case-fit c dt #:alpha (row-alpha row) #:l1-ratio r #:fit-intercept fi #:selection s))
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
      (twin-row c fi r 'cyclic)))
  (define fits
    (twin-results twin
                  (for/list ([row (in-list rows)])
                    (request "sklearn" row dt))))
  (define ran
    (for/sum ([row (in-list rows)] [t (in-list fits)])
             (match-define (struct* twin-row ([source c] [fit-intercept fi] [l1-ratio r])) row)
             (define p (problem-of (lcase-X c) (lcase-y c) (lcase-w c) fi))
             (define alpha (row-alpha row))
             (define o (oracle-elastic-net p alpha r))
             (match-define (struct* en-solution ([coef c*] [mu mu])) o)
             (define run (en-case-fit c dt #:alpha alpha #:l1-ratio r #:fit-intercept fi))
             (match-define (struct* en-run ([coef coef])) run)
             (define theirs-coef (map inexact->exact (hash-ref t 'coef)))
             (cond
               [(and (converged? run) (> mu 0))
                (test-case (format "~a ~a ~a l1-ratio ~a" (lcase-name c) dt fi r)
                  (check-law "sklearn-distance"
                             (coef-distance coef theirs-coef)
                             (+ (s2-bound o coef alpha r default-tol (unit-roundoff dt))
                                (coef-distance theirs-coef c*))))
                1]
               [else 0])))
  (test-case (format "sklearn-distance ran on every ~a fit" dt)
    (check-equal? ran (length rows))))
