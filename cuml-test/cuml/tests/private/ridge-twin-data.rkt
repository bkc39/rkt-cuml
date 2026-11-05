#lang racket/base

(require (only-in racket/match match-define)
         racket/runtime-path
         (only-in racket/sequence in-slice)
         (only-in rackunit check-equal? test-case)
         (only-in "linear-data.rkt"
                  check-law
                  exact-fit
                  floor-log2
                  gap
                  lcase
                  lcase-name
                  lcase-w
                  lcase-X
                  lcase-y
                  max-abs
                  normal-case
                  ridge-oracles
                  ridge-solution
                  unit-roundoff)
         (only-in "linear-twin-data.rkt" case-request theirs twin-results)
         (only-in "ridge-data.rkt" ridge-case-fit solvers widest-spread))

(provide check-sklearn-parity
         check-staged-parity
         check-raw-parity)

(define-runtime-path twin "../python/ridge_twin.py")

(define (request op c dt alpha solver fi)
  (case-request op
                c
                dt
                fi
                'alpha
                (real->double-flonum alpha)
                'solver
                (symbol->string solver)))

(define (results requests)
  (twin-results twin requests))

(define (headroom dt)
  (if (eq? dt 'float32) 16 64))

(define (staged-case c dt fi)
  (define sx (expt 2 (- (headroom dt) (quotient (floor-log2 (widest-spread c fi)) 2))))
  (define sy (expt 2 (- (floor-log2 (max-abs (lcase-y c))))))
  (lcase (format "staged ~a" (lcase-name c))
         (for/list ([row (in-list (lcase-X c))])
           (map (lambda (x) (* sx x)) row))
         (map (lambda (v) (* sy v)) (lcase-y c))
         (lcase-w c)))

(define staged-sources
  (list (normal-case 300 7 0)
        (normal-case 300 7 1 #:weighted? #t)
        (normal-case 300 2 2 #:weighted? #t)))

(define (check-staged-parity dt)
  (define rows
    (for*/list ([c (in-list staged-sources)]
                [fi (in-list '(#t #f))]
                [r (in-list (list (expt 2 -20) (expt 2 -7) 1))])
      (list (staged-case c dt fi) fi (* r (expt 2 (* 2 (headroom dt)))))))
  (define fits
    (results (for/list ([row (in-list rows)])
               (match-define (list c fi alpha) row)
               (request "cuml" c dt alpha 'eig fi))))
  (for ([row (in-list rows)]
        [t (in-list fits)])
    (match-define (list c fi alpha) row)
    (test-case (format "~a ~a ~a alpha ~a" (lcase-name c) dt fi alpha)
      (check-equal? (ridge-case-fit c dt #:alpha alpha #:solver 'eig #:fit-intercept fi)
                    (theirs dt t)))))

(define raw-sources
  (list (normal-case 200 7 0)
        (normal-case 200 7 1 #:weighted? #t)
        (normal-case 20 7 2)
        (normal-case 2000 7 4 #:weighted? #t)
        (normal-case 200 1 7)))

(define (raw-rows)
  (for*/list ([c (in-list raw-sources)]
              [fi (in-list '(#t #f))]
              [o (in-list
                  (ridge-oracles (lcase-X c) (lcase-y c) (lcase-w c) fi (list (expt 2 -7) 1)))])
    (list c fi o)))

(define (check-against dt rows references name)
  (for ([row (in-list rows)]
        [refs (in-list references)])
    (match-define (list c fi (ridge-solution alpha _ _ kappa _ _ _)) row)
    (define u (unit-roundoff dt))
    (for* ([ours (in-list solvers)]
           [reference (in-list refs)])
      (match-define (list label fit) reference)
      (test-case (format "~a ~a ~a alpha ~a ours ~a, ~a ~a"
                         (lcase-name c)
                         dt
                         fi
                         alpha
                         ours
                         name
                         label)
        (match-define (list coef _)
          (exact-fit (ridge-case-fit c dt #:alpha alpha #:solver ours #:fit-intercept fi)))
        (match-define (list coef* _) (exact-fit fit))
        (check-law "R4" (gap coef coef*) (* 256 u kappa (max-abs coef*)))))))

(define (python-solver c s)
  (match-define (cons first-row _) (lcase-X c))
  (if (and (eq? s 'eig) (= 1 (length first-row))) 'auto s))

(define (paired-references dt rows op labels solver-of fit-of)
  (define asked
    (for*/list ([row (in-list rows)]
                [s (in-list labels)])
      (match-define (list c fi (ridge-solution alpha _ _ _ _ _ _)) row)
      (request op c dt alpha (solver-of c s) fi)))
  (for/list ([pair (in-slice (length labels) (results asked))])
    (for/list ([label (in-list labels)]
               [r (in-list pair)])
      (list label (fit-of r)))))

(define (check-raw-parity dt)
  (define rows (raw-rows))
  (check-against dt
                 rows
                 (paired-references dt rows "cuml" '(eig svd) python-solver (lambda (r) (theirs dt r)))
                 "Python cuML"))

(define (check-sklearn-parity dt)
  (define rows (raw-rows))
  (check-against dt
                 rows
                 (paired-references dt
                                    rows
                                    "sklearn"
                                    '(svd cholesky)
                                    (lambda (_ s) s)
                                    (lambda (r) (list (hash-ref r 'coef) (hash-ref r 'intercept))))
                 "sklearn"))
