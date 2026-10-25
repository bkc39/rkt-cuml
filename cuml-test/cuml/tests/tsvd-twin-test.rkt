#lang racket/base

(require (only-in racket/list split-at)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in "private/pca-data.rkt" fixture-name)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/tsvd-data.rkt" accuracy-problems exact-reference)
         (only-in "private/tsvd-twin-data.rkt"
                  ours
                  rows-of
                  theirs-reference
                  twin-case
                  twin-cases
                  twin-results))

(define dtypes '(float32 float64))

(define (first-problems problems)
  (for/list ([p (in-list problems)]
             [_ (in-range 3)])
    p))

(test-twin "L8, L9: the fit is within c u F^2 of numpy's exact SVD and of sklearn's arpack fit"
  (define pairs
    (for*/list ([c (in-list twin-cases)]
                [dt (in-list dtypes)])
      (list c dt)))
  (define-values (exacts sklearns)
    (split-at (twin-results (for*/list ([op (in-list '("exact" "sklearn"))]
                                        [p (in-list pairs)])
                              (match-define (list c dt) p)
                              (list op c dt 'full)))
              (length pairs)))
  (for* ([(p exact sklearn) (in-parallel pairs exacts sklearns)]
         [solver (in-list '(full jacobi))])
    (match-define (list (and c (twin-case f k)) dt) p)
    (test-case (format "~a ~a ~a" (fixture-name f) dt solver)
      (define-values (rows fitted _T _I) (ours c dt solver))
      (match-define (cons first-row _) rows)
      (define d (length first-row))
      (match-define (hash-table ('singular_values spectrum) ('components components)) exact)
      (check-equal? (first-problems
                     (accuracy-problems fitted
                                        (exact-reference rows spectrum (rows-of components d) k)
                                        rows
                                        dt
                                        (if (eq? solver 'jacobi) 256 64)))
                    '()
                    "L8")
      (check-equal?
       (first-problems
        (accuracy-problems fitted (theirs-reference sklearn exact d) rows dt 128))
       '()
       "L9"))))
