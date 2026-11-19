#lang racket/base

(require (only-in cuml fit logistic-regression score)
         (only-in cuml/linear-model logistic-regression-fit logistic-regression-predict)
         (only-in datasets/core dataset-table-columns dataset-table-names load-dataset-table)
         (only-in racket/file file->value)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match-define struct*)
         racket/runtime-path
         (only-in rackunit check-equal? check-exn check-true test-case)
         (only-in threading ~>)
         (only-in "linear-data.rkt" columns-of on-matrix)
         (only-in "linear-twin-data.rkt" decoded)
         (only-in "logistic-data.rkt"
                  correct
                  cutoff-message
                  default-tol
                  default-tol-cap
                  fit-parts
                  gauged-fit
                  gauged-sklearn
                  lr-case
                  lr-case-X
                  lr-case-y
                  lr-fit
                  relative-gap
                  setting
                  setting-C
                  setting-l1-ratio
                  setting-penalty
                  tight-max-iter
                  tight-tol)
         (only-in "logistic-oracle.rkt" residual residual-of)
         (only-in "logistic-twin-data.rkt" logistic-twin logistic-twin-results setting-twin))

(provide (struct-out labelled)
         check-raw-optima
         check-raw-refusal
         check-raw-rescued
         check-score
         check-sklearn-optimum
         labelled-set
         raw-optima-path
         raw-optima-settings
         standardised
         tight-fit)

(struct labelled
        (case labels
          classes))

(define stationary-bound 1e-7)
(define raw-stationary-bound 1e-5)
(struct gap-check (fits bound))
(define float32-l2-median-gap-bound 1.6e-4)
(define float32-gap-checks
  (hash 'l1 (gap-check 1 2.9e-2)
        'l2 (gap-check 5 float32-l2-median-gap-bound)
        'elasticnet (gap-check 1 1.7e-3)))
(define raw-optima-settings
  (list (setting 'l1 1.0 #f) (setting 'elasticnet 1.0 0.5) (setting 'l2 0.01 #f)))
(define-runtime-path raw-optima-path "raw-logistic-optima.rktd")
(define sklearn-gap-bound 1e-5)
(define row-slack 1)

(define (flonum-exact v)
  (inexact->exact (real->double-flonum v)))

(define (standardised X)
  (define (scaled xs)
    (define n (length xs))
    (define m (/ (apply + xs) n))
    (define s
      (~> (for/sum ([x (in-list xs)]) (expt (- x m) 2)) (/ n) exact->inexact sqrt inexact->exact))
    (map (lambda (x) (flonum-exact (/ (- x m) s))) xs))
  (~> X columns-of (map scaled _) columns-of))

(define (labelled-set name response excluded #:standardise? [standardise? #f])
  (define table (load-dataset-table name))
  (define named
    (for/hash ([n (in-list (dataset-table-names table))]
               [column (in-vector (dataset-table-columns table))])
      (values n (vector->list column))))
  (define predictors (remove* (cons response excluded) (dataset-table-names table)))
  (define X
    (columns-of (for/list ([p (in-list predictors)])
                  (map flonum-exact (hash-ref named p)))))
  (define labels (hash-ref named response))
  (define classes (sort (remove-duplicates labels) string<?))
  (define codes
    (for/hash ([c (in-list classes)]
               [i (in-naturals)])
      (values c i)))
  (labelled (lr-case (format "~a" name)
                     (if standardise?
                         (standardised X)
                         X)
                     (map (lambda (l) (hash-ref codes l)) labels)
                     #f)
            labels
            classes))

(define (tight-fit set dt s)
  (logistic-regression-fit (on-matrix (lr-case-X (labelled-case set)) dt)
                           (labelled-labels set)
                           #:penalty (setting-penalty s)
                           #:C (setting-C s)
                           #:l1-ratio (setting-l1-ratio s)
                           #:tol (tight-tol dt)
                           #:max-iter tight-max-iter))

(define (sklearn-labels set theirs)
  (define classes (list->vector (labelled-classes set)))
  (map (lambda (i) (vector-ref classes i)) (hash-ref theirs 'predict)))

(define (check-optimum set dt s parts #:stationary [stationary stationary-bound])
  (match-define (struct* labelled ((case c))) set)
  (match-define (struct* lr-fit ([coef coef] [intercept intercept])) parts)
  (cond
    [(eq? dt 'float64)
     (match-define (residual K F _)
       (residual-of (lr-case-X c)
                    (lr-case-y c)
                    #f
                    coef
                    intercept
                    #:penalty (setting-penalty s)
                    #:C (setting-C s)
                    #:l1-ratio (setting-l1-ratio s)
                    #:tol (tight-tol 'float64)))
     (check-true (<= K (* stationary (max F 1)))
                 (format "K ~a F ~a" (exact->inexact K) (exact->inexact F)))]
    [else
     (match-define (gap-check fits bound) (hash-ref float32-gap-checks (setting-penalty s)))
     (define optimum (gauged-fit (fit-parts (tight-fit set 'float64 s))))
     (define repeats
       (for/list ([_ (in-range (sub1 fits))])
         (fit-parts (tight-fit set dt s))))
     (define gaps (map (lambda (f) (relative-gap (gauged-fit f) optimum)) (cons parts repeats)))
     (define gap (list-ref (sort gaps <) (quotient fits 2)))
     (check-true (<= gap bound)
                 (format "median gap to the float64 fit of ~a fits ~a, bound ~a" fits gap bound))]))

(struct asked (setting tol) #:transparent)

(define (check-sklearn-optimum set dt settings)
  (match-define (struct* labelled ((case c) [labels labels])) set)
  (define (l2? s)
    (eq? (setting-penalty s) 'l2))
  (define (tight? s)
    (and (eq? dt 'float64) (l2? s)))
  (define keys
    (for*/list ([s (in-list settings)]
                [tol (in-list (if (tight? s)
                                  (list default-tol (tight-tol 'float64))
                                  (list default-tol)))])
      (asked s tol)))
  (define (request key)
    (match-define (asked s tol) key)
    (setting-twin "sklearn-fit" c 'float64 s #:tol tol))
  (define answers
    (for/hash ([key (in-list keys)]
               [theirs (in-list (logistic-twin-results (map request keys)))])
      (values key theirs)))
  (for ([s (in-list settings)])
    (test-case (format "~a ~a" dt (setting-penalty s))
      (define r (tight-fit set dt s))
      (check-optimum set dt s (fit-parts r))
      (when (tight? s)
        (define gap
          (relative-gap (gauged-fit (fit-parts r))
                        (gauged-sklearn (hash-ref answers (asked s (tight-tol 'float64))))))
        (check-true (<= gap sklearn-gap-bound) (format "coef gap to sklearn ~a" gap)))
      (define ours (correct (logistic-regression-predict r (on-matrix (lr-case-X c) dt)) labels))
      (define sklearn (correct (sklearn-labels set (hash-ref answers (asked s default-tol))) labels))
      (check-true (<= (abs (- ours sklearn)) row-slack)
                  (format "correct ~a, sklearn ~a" ours sklearn)))))

(define (python-and-oracle c dt)
  (logistic-twin-results (list (logistic-twin "cuml-fit" c dt)
                               (logistic-twin "sklearn-fit" c 'float64 #:tol (tight-tol 'float64)))))

(define (check-python-off-optimum python oracle dt)
  (define gap
    (relative-gap (append (decoded dt (hash-ref python 'coef_bits))
                          (decoded dt (hash-ref python 'intercept_bits)))
                  (gauged-sklearn oracle)))
  (check-true (> gap (default-tol-cap dt))
              (format "Python cuML's point: coef gap ~a (cap ~a)" gap (default-tol-cap dt))))

(define (check-raw-refusal set dt)
  (match-define (struct* labelled ((case c) [labels labels])) set)
  (check-exn cutoff-message (lambda () (logistic-regression-fit (on-matrix (lr-case-X c) dt) labels)))
  (match-define (list python oracle) (python-and-oracle c dt))
  (check-python-off-optimum python oracle dt))

(define (check-raw-rescued set capped refused)
  (match-define (struct* labelled ((case c) [labels labels])) set)
  (define (raw-fit dt s max-iter)
    (logistic-regression-fit (on-matrix (lr-case-X c) dt)
                             labels
                             #:penalty (setting-penalty s)
                             #:C (setting-C s)
                             #:l1-ratio (setting-l1-ratio s)
                             #:max-iter max-iter))
  (define (refused-fit e)
    (unless (regexp-match? cutoff-message (exn-message e))
      (raise e))
    #f)
  (define optima (raw-optima))
  (for ([arm (in-list capped)])
    (match-define (list dt penalty C l1-ratio max-iter outcome cap) arm)
    (define s (setting penalty C l1-ratio))
    (test-case (format "~a near the optimum ~a" outcome arm)
      (define optimum (hash-ref optima s))
      (define r
        (with-handlers ([exn:fail? refused-fit])
          (raw-fit dt s max-iter)))
      (define gap (and r (relative-gap (gauged-fit (fit-parts r)) (gauged-fit optimum))))
      (check-true (if gap
                      (<= gap cap)
                      (eq? outcome 'either))
                  (if gap
                      (format "coef gap to the optimum ~a (cap ~a)" gap cap)
                      "refused, but this arm has only ever been accepted"))))
  (for ([arm (in-list refused)])
    (match-define (list dt penalty C l1-ratio max-iter) arm)
    (test-case (format "refused ~a" arm)
      (check-exn cutoff-message (lambda () (raw-fit dt (setting penalty C l1-ratio) max-iter))))))

(define (raw-optima)
  (for/hash ([entry (in-list (file->value raw-optima-path))])
    (match-define (list (list penalty C l1-ratio) coef intercept) entry)
    (values (setting penalty C l1-ratio) (lr-fit coef intercept #f))))

(define (check-raw-optima set)
  (define optima (raw-optima))
  (for ([s (in-list raw-optima-settings)])
    (test-case (format "the pinned raw optimum ~a is stationary" s)
      (check-optimum set 'float64 s (hash-ref optima s) #:stationary raw-stationary-bound))))

(define (check-score set)
  (match-define (struct* labelled ((case c) [labels labels])) set)
  (define X (on-matrix (lr-case-X c) 'float64))
  (match-define (list theirs)
    (logistic-twin-results
     (list (logistic-twin "sklearn-fit" c 'float64 #:tol (tight-tol 'float64)))))
  (define e (fit (logistic-regression #:tol (tight-tol 'float64) #:max-iter tight-max-iter) X labels))
  (check-equal? (score e X labels)
                (exact->inexact (/ (correct (sklearn-labels set theirs) labels) (length labels)))))
