#lang racket/base

(require (only-in racket/list split-at)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in "private/pca-data.rkt" fixture-name)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/scaler-data.rkt" rows-of)
         (only-in "private/tsvd-data.rkt"
                  accuracy-problems
                  exact-reference
                  extents
                  first-problems
                  sign-matched
                  tie-fixture)
         (only-in "private/tsvd-twin-data.rkt"
                  ours
                  theirs-reference
                  twin-case
                  twin-cases
                  twin-results))

(define dtypes '(float32 float64))

(define (compared f ref fitted)
  (if (eq? f tie-fixture)
      (sign-matched ref fitted)
      ref))

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
      (define-values (_n d) (extents rows))
      (match-define (hash-table ('singular_values spectrum) ('components components)) exact)
      (check-equal?
       (first-problems (accuracy-problems
                        fitted
                        (compared f (exact-reference rows spectrum (rows-of components d) k) fitted)
                        rows
                        dt
                        (if (eq? solver 'jacobi) 256 64)))
       '()
       "L8")
      (check-equal?
       (first-problems
        (accuracy-problems fitted (compared f (theirs-reference sklearn exact d) fitted) rows dt 128))
       '()
       "L9"))))
