#lang racket/base

(require (only-in cuml fit logistic-regression score)
         (only-in cuml/linear-model logistic-regression-fit logistic-regression-predict)
         (only-in datasets/core dataset-table-columns dataset-table-names load-dataset-table)
         (only-in racket/file file->value)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match match-define struct*)
         racket/runtime-path
         (only-in rackunit check-eq? check-equal? check-exn check-true test-case)
         (only-in threading ~>)
         (only-in "linear-data.rkt" columns-of on-matrix)
         (only-in "linear-twin-data.rkt" decoded)
         (only-in "logistic-data.rkt"
                  correct
                  cutoff-exn?
                  default-tol
                  default-tol-cap
                  fit-parts
                  gauged-fit
                  gauged-sklearn
                  lr-case
                  lr-case-X
                  lr-fit
                  relative-gap
                  setting
                  setting-C
                  setting-l1-ratio
                  setting-penalty
                  tight-max-iter
                  tight-tol)
         (only-in "logistic-oracle.rkt" case-residual residual residual-bound residual-gradient)
         (only-in "logistic-twin-data.rkt" logistic-twin logistic-twin-results setting-twin))

(provide (struct-out labelled)
         check-raw-optima
         check-raw-refusal
         check-raw-rescued
         check-score
         check-sklearn-optimum
         check-stationary
         labelled-set
         raw-optima-path
         raw-optima-settings
         residual-at
         standardised
         stationary-bound
         tight-fit)

(struct labelled
        (case labels
          classes))

(define stationary-bound 1e-7)
(define raw-stationary-bound 1e-5)
(struct gap-check (fits statistic bound))
(define statistic-ranks (hash 'smallest 0 'second-smallest 1))
(define float32-gap-checks
  (hash 'l1 (gap-check 1 'smallest 2.9e-2)
        'l2 (gap-check 5 'second-smallest 1.6e-4)
        'elasticnet (gap-check 1 'smallest 1.7e-3)))
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
  (cond
    [(eq? dt 'float64)
     (check-stationary (case-residual c parts s #:tol (tight-tol 'float64)) stationary)]
    [else
     (match-define (struct* gap-check ([fits fits] [statistic statistic] [bound bound]))
       (hash-ref float32-gap-checks (setting-penalty s)))
     (define optimum (gauged-fit (fit-parts (tight-fit set 'float64 s))))
     (define repeats
       (for/list ([_ (in-range (sub1 fits))])
         (fit-parts (tight-fit set dt s))))
     (define gaps (map (lambda (f) (relative-gap (gauged-fit f) optimum)) (cons parts repeats)))
     (define gap (list-ref (sort gaps <) (hash-ref statistic-ranks statistic)))
     (check-true
      (<= gap bound)
      (format "~a gap of ~a fits to the float64 fit ~a, bound ~a" statistic fits gap bound))]))

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
  (check-exn cutoff-exn? (lambda () (logistic-regression-fit (on-matrix (lr-case-X c) dt) labels)))
  (match-define (list python oracle) (python-and-oracle c dt))
  (check-python-off-optimum python oracle dt))

(struct arm (dtype setting max-iter outcome cap) #:transparent)

(define (arm-of row)
  (match row
    [(list dt (list penalty C l1-ratio) max-iter outcome cap)
     (arm dt (setting penalty C l1-ratio) max-iter outcome cap)]
    [(list dt (list penalty C l1-ratio) max-iter)
     (arm dt (setting penalty C l1-ratio) max-iter 'refused #f)]))

(define (arm-name a)
  (match-define (struct* arm ([dtype dt] [setting s] [max-iter max-iter] [outcome outcome])) a)
  (match-define (setting penalty C l1-ratio) s)
  (format "~a ~a C ~a l1-ratio ~a at max-iter ~a: ~a" dt penalty C l1-ratio max-iter outcome))

(define (raw-fit set a)
  (match-define (struct* labelled ((case c) [labels labels])) set)
  (match-define (struct* arm ([dtype dt] [setting s] [max-iter max-iter])) a)
  (logistic-regression-fit (on-matrix (lr-case-X c) dt)
                           labels
                           #:penalty (setting-penalty s)
                           #:C (setting-C s)
                           #:l1-ratio (setting-l1-ratio s)
                           #:max-iter max-iter))

(define (residual-at r X y #:penalty [penalty 'l2] #:fit-intercept [fi #t])
  (case-residual (lr-case "residual-at" X y #f)
                 (fit-parts r)
                 (setting penalty 1.0 #f)
                 #:fit-intercept fi))

(define (check-stationary res [stationary stationary-bound])
  (match-define (residual K F _) res)
  (check-true (<= K (* stationary (max F 1)))
              (format "K ~a F ~a" (exact->inexact K) (exact->inexact F))))

(define (check-capped-arm set optima a)
  (match-define (struct* labelled ((case c))) set)
  (match-define (struct* arm ([dtype dt] [setting s] [outcome outcome] [cap cap])) a)
  (test-case (arm-name a)
    (define r
      (with-handlers ([cutoff-exn? (lambda (_) #f)])
        (raw-fit set a)))
    (define gap (and r (relative-gap (gauged-fit (fit-parts r)) (gauged-fit (hash-ref optima s)))))
    (define res (and r (not cap) (case-residual c (fit-parts r) s #:dtype dt)))
    (cond
      [(not gap) (check-eq? outcome 'either "refused, but this arm has only ever been accepted")]
      [cap (check-true (<= gap cap) (format "coef gap to the optimum ~a (cap ~a)" gap cap))]
      [else
       (check-true (<= (residual-gradient res) (residual-bound res))
                   (format "accepted at residual ~a, bound ~a"
                           (exact->inexact (residual-gradient res))
                           (exact->inexact (residual-bound res))))])))

(define (check-refused-arm set a)
  (test-case (arm-name a)
    (check-exn cutoff-exn? (lambda () (raw-fit set a)))))

(define (check-raw-rescued set capped refused)
  (define optima (raw-optima))
  (for ([row (in-list capped)])
    (check-capped-arm set optima (arm-of row)))
  (for ([row (in-list refused)])
    (check-refused-arm set (arm-of row))))

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
