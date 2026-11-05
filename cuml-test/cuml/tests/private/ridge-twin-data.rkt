#lang racket/base

(require (only-in cuml/linear-model/private/native libcuml-version)
         (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-equal? test-case)
         (only-in threading ~>)
         (only-in "linear-data.rkt"
                  check-law
                  exact-fit
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
                  unit-roundoff
                  weights-or-ones)
         (only-in "linear-twin-data.rkt" bits-of decoded)
         (only-in "python-env.rkt" run-twin)
         (only-in "ridge-data.rkt" ridge-case-fit solvers))

(provide check-sklearn-parity
         check-staged-parity
         check-raw-parity)

(define-runtime-path twin "../python/ridge_twin.py")

(define (request op c dt alpha solver fi)
  (hasheq 'op op
          'dtype (symbol->string dt)
          'X (bits-of dt (~> c lcase-X (apply append _) (map exact->inexact _)))
          'y (bits-of dt (map exact->inexact (lcase-y c)))
          'w (if (lcase-w c)
                 (bits-of dt (map exact->inexact (lcase-w c)))
                 'null)
          'alpha (real->double-flonum alpha)
          'solver (symbol->string solver)
          'fit_intercept fi))

(define (results requests)
  (match-define (hash-table ('results results) ('version (hash-table ('cuml version))))
    (run-twin twin (hasheq 'cases requests)))
  (check-equal? version libcuml-version)
  results)

(define (cuml-fit dt result)
  (match-define (list intercept) (decoded dt (hash-ref result 'intercept_bits)))
  (list (decoded dt (hash-ref result 'coef_bits)) intercept))

(define (headroom dt)
  (if (eq? dt 'float32) 16 64))

(define (floor-log2 x)
  (let loop ([e (- (integer-length (numerator x)) (integer-length (denominator x)))])
    (cond
      [(> (expt 2 e) x) (loop (sub1 e))]
      [(<= (expt 2 (add1 e)) x) (loop (add1 e))]
      [else e])))

(define (staged-case c dt fi)
  (define weights (weights-or-ones (lcase-w c) (lcase-y c)))
  (define total (apply + weights))
  (define (spread2 col)
    (define m
      (if fi
          (/ (for/sum ([x (in-list col)] [w (in-list weights)]) (* w x)) total)
          0))
    (for/sum ([x (in-list col)] [w (in-list weights)]) (* w (expt (- x m) 2))))
  (define widest (apply max (map spread2 (apply map list (lcase-X c)))))
  (define sx (expt 2 (- (headroom dt) (quotient (floor-log2 widest) 2))))
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
  (define theirs
    (results (for/list ([row (in-list rows)])
               (match-define (list c fi alpha) row)
               (request "cuml" c dt alpha 'eig fi))))
  (for ([row (in-list rows)]
        [t (in-list theirs)])
    (match-define (list c fi alpha) row)
    (test-case (format "~a ~a ~a alpha ~a" (lcase-name c) dt fi alpha)
      (check-equal? (ridge-case-fit c dt #:alpha alpha #:solver 'eig #:fit-intercept fi)
                    (cuml-fit dt t)))))

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

(define (check-against dt rows theirs-of name)
  (for ([row (in-list rows)])
    (match-define (list c fi (ridge-solution alpha _ _ kappa _ _ _)) row)
    (define u (unit-roundoff dt))
    (for* ([ours (in-list solvers)]
           [reference (in-list (theirs-of row))])
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
  (if (and (eq? s 'eig) (= 1 (length (car (lcase-X c))))) 'auto s))

(define (check-raw-parity dt)
  (define rows (raw-rows))
  (define asked
    (for*/list ([row (in-list rows)]
                [s (in-list '(eig svd))])
      (match-define (list c fi (ridge-solution alpha _ _ _ _ _ _)) row)
      (request "cuml" c dt alpha (python-solver c s) fi)))
  (define theirs (results asked))
  (define by-row
    (for/list ([i (in-range 0 (length theirs) 2)])
      (list (list 'eig (cuml-fit dt (list-ref theirs i)))
            (list 'svd (cuml-fit dt (list-ref theirs (add1 i)))))))
  (define table (map cons rows by-row))
  (check-against dt rows (lambda (row) (cdr (assq row table))) "Python cuML"))

(define (check-sklearn-parity dt)
  (define rows (raw-rows))
  (define asked
    (for*/list ([row (in-list rows)]
                [s (in-list '(svd cholesky))])
      (match-define (list c fi (ridge-solution alpha _ _ _ _ _ _)) row)
      (request "sklearn" c dt alpha s fi)))
  (define theirs (results asked))
  (define (sk r)
    (list (hash-ref r 'coef) (hash-ref r 'intercept)))
  (define table
    (for/list ([row (in-list rows)]
               [i (in-naturals)])
      (list row
            (list 'svd (sk (list-ref theirs (* 2 i))))
            (list 'cholesky (sk (list-ref theirs (add1 (* 2 i))))))))
  (check-against dt rows (lambda (row) (cdr (assq row table))) "sklearn"))
