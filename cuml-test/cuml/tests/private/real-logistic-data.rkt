#lang racket/base

(require (only-in cuml fit logistic-regression score)
         (only-in cuml/linear-model logistic-regression-fit logistic-regression-predict)
         (only-in datasets/core dataset-table-columns dataset-table-names load-dataset-table)
         (only-in racket/list index-of remove-duplicates)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-exn check-true test-case)
         (only-in threading ~>)
         (only-in "linear-data.rkt" columns-of on-matrix)
         (only-in "linear-twin-data.rkt" decoded logistic-twin logistic-twin-results)
         (only-in "logistic-check.rkt" residual residual-of)
         (only-in "logistic-data.rkt" centred fit-parts lr-case lr-case-X lr-case-y relative-gap))

(provide (struct-out labelled)
         check-raw-refusal
         check-score
         check-sklearn-optimum
         labelled-set)

(struct labelled
        (case labels
          classes))

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
  (labelled (lr-case (format "~a" name)
                     (if standardise?
                         (standardised X)
                         X)
                     (map (lambda (l) (index-of classes l)) labels)
                     #f)
            labels
            classes))

(define (tight-fit set dt penalty C ratio)
  (logistic-regression-fit (on-matrix (lr-case-X (labelled-case set)) dt)
                           (labelled-labels set)
                           #:penalty penalty
                           #:C C
                           #:l1-ratio ratio
                           #:tol (if (eq? dt 'float32) 1e-6 1e-10)
                           #:max-iter 100000))

(define (gauged coef intercept)
  (if (= 1 (length coef))
      (append (car coef) intercept)
      (append (apply append (centred coef)) (centred intercept))))

(define (correct predicted truth)
  (for/sum ([p (in-list predicted)] [t (in-list truth)]) (if (equal? p t) 1 0)))

(define (sklearn-labels set theirs)
  (map (lambda (i) (list-ref (labelled-classes set) i)) (hash-ref theirs 'predict)))

(define (sklearn-request set penalty C ratio tol)
  (logistic-twin "sklearn-fit"
                 (labelled-case set)
                 'float64
                 (hasheq 'penalty (symbol->string penalty) 'C C 'l1_ratio (or ratio 'null) 'tol tol)))

(define (check-optimum set dt penalty C ratio r)
  (match-define (labelled c _ _) set)
  (match-define (list coef intercept _) (fit-parts r))
  (cond
    [(eq? dt 'float64)
     (match-define (residual K F _)
       (residual-of (lr-case-X c)
                    (lr-case-y c)
                    #f
                    coef
                    intercept
                    #:penalty penalty
                    #:C C
                    #:l1-ratio ratio
                    #:tol 1e-10))
     (check-true (<= K (* 1e-7 (max F 1)))
                 (format "K ~a F ~a" (exact->inexact K) (exact->inexact F)))]
    [else
     (match-define (list exact-coef exact-intercept _)
       (fit-parts (tight-fit set 'float64 penalty C ratio)))
     (define gap (relative-gap (gauged coef intercept) (gauged exact-coef exact-intercept)))
     (check-true (<= gap 1e-2) (format "gap to the float64 fit ~a" gap))]))

(define (check-sklearn-optimum set dt penalties)
  (match-define (labelled c labels _) set)
  (define (l2? s)
    (eq? (car s) 'l2))
  (define requests
    (append (for/list ([s (in-list penalties)])
              (apply sklearn-request set (append s '(1e-4))))
            (for/list ([s (in-list penalties)]
                       #:when (and (eq? dt 'float64) (l2? s)))
              (apply sklearn-request set (append s '(1e-10))))))
  (define results (logistic-twin-results requests))
  (define tight (list-tail results (length penalties)))
  (for ([s (in-list penalties)]
        [theirs (in-list results)])
    (match-define (list penalty C ratio) s)
    (test-case (format "~a ~a" dt penalty)
      (define r (tight-fit set dt penalty C ratio))
      (check-optimum set dt penalty C ratio r)
      (when (and (eq? dt 'float64) (l2? s))
        (match-define (list coef intercept _) (fit-parts r))
        (define sk (car tight))
        (define gap
          (relative-gap (gauged coef intercept)
                        (gauged (hash-ref sk 'coef) (hash-ref sk 'intercept))))
        (check-true (<= gap 1e-5) (format "coef gap to sklearn ~a" gap)))
      (define ours (correct (logistic-regression-predict r (on-matrix (lr-case-X c) dt)) labels))
      (define sklearn (correct (sklearn-labels set theirs) labels))
      (check-true (<= (abs (- ours sklearn)) 1) (format "correct ~a, sklearn ~a" ours sklearn)))))

(define cutoff-message #rx"first-order residual .* above the bound")

(define (s4-cap dt)
  (if (eq? dt 'float32) 8e-2 3e-2))

(define (check-raw-refusal set dt)
  (match-define (labelled c labels _) set)
  (check-exn cutoff-message (lambda () (logistic-regression-fit (on-matrix (lr-case-X c) dt) labels)))
  (match-define (list python oracle)
    (logistic-twin-results (list (logistic-twin "cuml-fit" c dt (hasheq))
                                 (logistic-twin "sklearn-fit" c 'float64 (hasheq 'tol 1e-10)))))
  (define coef (decoded dt (hash-ref python 'coef_bits)))
  (define intercept (decoded dt (hash-ref python 'intercept_bits)))
  (define gap
    (relative-gap (append coef intercept)
                  (gauged (hash-ref oracle 'coef) (hash-ref oracle 'intercept))))
  (match-define (residual K _ bound)
    (residual-of (lr-case-X c) (lr-case-y c) #f (list coef) intercept))
  (check-true
   (or (> gap (s4-cap dt)) (> K bound))
   (format "Python cuML's point: coef gap ~a (cap ~a), K ~a (bound ~a)" gap (s4-cap dt) K bound)))

(define (check-score set)
  (match-define (labelled c labels _) set)
  (define X (on-matrix (lr-case-X c) 'float64))
  (match-define (list theirs)
    (logistic-twin-results (list (logistic-twin "sklearn-fit" c 'float64 (hasheq 'tol 1e-10)))))
  (define e (fit (logistic-regression #:tol 1e-10 #:max-iter 100000) X labels))
  (check-equal? (score e X labels)
                (exact->inexact (/ (correct (sklearn-labels set theirs) labels) (length labels)))))
