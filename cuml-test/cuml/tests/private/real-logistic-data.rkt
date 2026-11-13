#lang racket/base

(require (only-in cuml fit logistic-regression score)
         (only-in cuml/linear-model logistic-regression-fit logistic-regression-predict)
         (only-in datasets/core dataset-table-columns dataset-table-names load-dataset-table)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match-define struct*)
         (only-in rackunit check-equal? check-exn check-true test-case)
         (only-in threading ~>)
         (only-in "linear-data.rkt" columns-of on-matrix)
         (only-in "linear-twin-data.rkt" decoded)
         (only-in "logistic-data.rkt"
                  cutoff-message
                  default-tol-cap
                  fit-parts
                  gauged
                  lr-case
                  lr-case-X
                  lr-case-y
                  lr-fit
                  relative-gap
                  setting-C
                  setting-l1-ratio
                  setting-penalty
                  tight-max-iter
                  tight-tol)
         (only-in "logistic-oracle.rkt" residual residual-of)
         (only-in "logistic-twin-data.rkt" logistic-twin logistic-twin-results setting-twin))

(provide (struct-out labelled)
         check-raw-refusal
         check-score
         check-sklearn-optimum
         labelled-set)

(struct labelled
        (case labels
          classes))

(define stationary-bound 1e-7)
(define float32-gap-bound 1e-2)
(define sklearn-gap-bound 1e-5)
(define row-slack 1)
(define default-tol 1e-4)

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

(define (gauged-fit f)
  (match-define (struct* lr-fit ([coef coef] [intercept intercept])) f)
  (gauged coef intercept))

(define (gauged-sklearn theirs)
  (gauged (hash-ref theirs 'coef) (hash-ref theirs 'intercept)))

(define (correct predicted truth)
  (for/sum ([p (in-list predicted)] [t (in-list truth)]) (if (equal? p t) 1 0)))

(define (sklearn-labels set theirs)
  (define classes (list->vector (labelled-classes set)))
  (map (lambda (i) (vector-ref classes i)) (hash-ref theirs 'predict)))

(define (check-optimum set dt s r)
  (match-define (struct* labelled ((case c))) set)
  (match-define (struct* lr-fit ([coef coef] [intercept intercept])) (fit-parts r))
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
     (check-true (<= K (* stationary-bound (max F 1)))
                 (format "K ~a F ~a" (exact->inexact K) (exact->inexact F)))]
    [else
     (define gap
       (relative-gap (gauged-fit (fit-parts r)) (gauged-fit (fit-parts (tight-fit set 'float64 s)))))
     (check-true (<= gap float32-gap-bound) (format "gap to the float64 fit ~a" gap))]))

(define (check-sklearn-optimum set dt settings)
  (match-define (struct* labelled ((case c) [labels labels])) set)
  (define (l2? s)
    (eq? (setting-penalty s) 'l2))
  (define (tight? s)
    (and (eq? dt 'float64) (l2? s)))
  (define asked
    (for*/list ([s (in-list settings)]
                [tol (in-list (if (tight? s)
                                  (list default-tol (tight-tol 'float64))
                                  (list default-tol)))])
      (cons s tol)))
  (define answers
    (for/hash ([key (in-list asked)]
               [theirs (in-list
                        (logistic-twin-results
                         (for/list ([key (in-list asked)])
                           (setting-twin "sklearn-fit" c 'float64 (car key) #:tol (cdr key)))))])
      (values key theirs)))
  (for ([s (in-list settings)])
    (test-case (format "~a ~a" dt (setting-penalty s))
      (define r (tight-fit set dt s))
      (check-optimum set dt s r)
      (when (tight? s)
        (define gap
          (relative-gap (gauged-fit (fit-parts r))
                        (gauged-sklearn (hash-ref answers (cons s (tight-tol 'float64))))))
        (check-true (<= gap sklearn-gap-bound) (format "coef gap to sklearn ~a" gap)))
      (define ours (correct (logistic-regression-predict r (on-matrix (lr-case-X c) dt)) labels))
      (define sklearn (correct (sklearn-labels set (hash-ref answers (cons s default-tol))) labels))
      (check-true (<= (abs (- ours sklearn)) row-slack)
                  (format "correct ~a, sklearn ~a" ours sklearn)))))

(define (check-raw-refusal set dt)
  (match-define (struct* labelled ((case c) [labels labels])) set)
  (check-exn cutoff-message (lambda () (logistic-regression-fit (on-matrix (lr-case-X c) dt) labels)))
  (match-define (list python oracle)
    (logistic-twin-results
     (list (logistic-twin "cuml-fit" c dt)
           (logistic-twin "sklearn-fit" c 'float64 #:tol (tight-tol 'float64)))))
  (define gap
    (relative-gap (append (decoded dt (hash-ref python 'coef_bits))
                          (decoded dt (hash-ref python 'intercept_bits)))
                  (gauged-sklearn oracle)))
  (check-true (> gap (default-tol-cap dt))
              (format "Python cuML's point: coef gap ~a (cap ~a)" gap (default-tol-cap dt))))

(define (check-score set)
  (match-define (struct* labelled ((case c) [labels labels])) set)
  (define X (on-matrix (lr-case-X c) 'float64))
  (match-define (list theirs)
    (logistic-twin-results
     (list (logistic-twin "sklearn-fit" c 'float64 #:tol (tight-tol 'float64)))))
  (define e (fit (logistic-regression #:tol (tight-tol 'float64) #:max-iter tight-max-iter) X labels))
  (check-equal? (score e X labels)
                (exact->inexact (/ (correct (sklearn-labels set theirs) labels) (length labels)))))
