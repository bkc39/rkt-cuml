#lang racket/base

(require (only-in cuml/linear-model
                  logistic-regression-decision-function
                  logistic-regression-fit
                  logistic-regression-predict
                  logistic-regression-predict-proba)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in "private/linear-data.rkt" dtypes on-matrix on-vector unit-roundoff)
         (only-in "private/linear-twin-data.rkt" decoded logistic-twin logistic-twin-results)
         (only-in "private/logistic-data.rkt"
                  blob-case
                  case-fit
                  fit-parts
                  lr-case
                  lr-case-name
                  lr-case-w
                  lr-case-X
                  lr-case-y
                  relative-gap)
         (only-in "private/python-env.rkt" test-twin))

(define cases
  (list (blob-case "binary" 150 4 2 21)
        (blob-case "three classes" 150 4 3 22)
        (blob-case "five classes" 200 4 5 23)))

(define settings '((l2 1.0 #f) (l1 1.0 #f) (elasticnet 0.5 0.5) (none 1.0 #f)))

(define rows
  (for*/list ([dt (in-list dtypes)]
              [c (in-list cases)]
              [s (in-list settings)]
              [fi (in-list '(#t #f))])
    (list dt c s fi)))

(define (request row #:op [op "cuml-apply"])
  (match-define (list dt c (list penalty C ratio) fi) row)
  (logistic-twin
   op
   c
   dt
   (hasheq 'penalty (symbol->string penalty) 'C C 'l1_ratio (or ratio 'null) 'fit_intercept fi)))

(define summing-to-n
  (let* ([c (blob-case "binary weighted" 150 4 2 24 #:weighted? #t)]
         [w (lr-case-w c)]
         [scale (/ (length w) (apply + w))])
    (lr-case "binary weights summing to n"
             (lr-case-X c)
             (lr-case-y c)
             (map (lambda (v) (* v scale)) w))))

(define (tight-tol dt)
  (if (eq? dt 'float32) 1e-6 1e-10))

(define (tau-bound dt)
  (if (eq? dt 'float32)
      0
      (min (* 64 1e-6) 3e-2)))

(test-twin "with weights summing to n, Python cuML's fit in float32 and its optimum within 64 tau_T in float64 (T2)"
  (define settings* '((l2 1.0 #f) (l1 1.0 #f) (elasticnet 0.5 0.5)))
  (define requests
    (for*/list ([dt (in-list dtypes)]
                [s (in-list settings*)])
      (match-define (list penalty C ratio) s)
      (logistic-twin "cuml-fit"
                     summing-to-n
                     dt
                     (hasheq 'penalty (symbol->string penalty)
                             'C C
                             'l1_ratio (or ratio 'null)
                             'tol (tight-tol dt)
                             'max_iter 100000))))
  (for ([g (in-list (for*/list ([dt (in-list dtypes)]
                                [s (in-list settings*)])
                      (cons dt s)))]
        [theirs (in-list (logistic-twin-results requests))])
    (match-define (list dt penalty C ratio) g)
    (test-case (format "~a ~a" dt penalty)
      (match-define (list coef intercept _)
        (fit-parts (case-fit summing-to-n
                             dt
                             #:penalty penalty
                             #:C C
                             #:l1-ratio ratio
                             #:tol (tight-tol dt)
                             #:max-iter 100000)))
      (define gap
        (relative-gap (append (car coef) intercept)
                      (append (decoded dt (hash-ref theirs 'coef_bits))
                              (decoded dt (hash-ref theirs 'intercept_bits)))))
      (check-true (<= gap (tau-bound dt)) (format "gap ~a" gap)))))

(define (flat m)
  (apply append (device-matrix->list* m)))

(define (classes-of c)
  (length (remove-duplicates (lr-case-y c))))

(define (proba-gap ours theirs k)
  (for/fold ([worst 0.0])
            ([o (in-list ours)]
             [t (in-list theirs)])
    (max worst
         (/ (abs (- o t))
            (if (= k 2)
                1.0
                (* (+ k 3) t (+ 1 (abs (log t)))))))))

(define (their-parts theirs dt)
  (list (decoded dt (hash-ref theirs 'coef_bits))
        (decoded dt (hash-ref theirs 'intercept_bits))
        (hash-ref theirs 'n_iter)))

(define (flat-parts r)
  (match-define (list coef intercept n-iter) (fit-parts r))
  (list (apply append coef) intercept n-iter))

(define (repeatable? row)
  (match-define (list dt c (list penalty _ _) _) row)
  (not (and (eq? dt 'float32) (> (classes-of c) 2) (memq penalty '(l1 elasticnet)))))

(define (row-fit row X)
  (match-define (list _ c (list penalty C ratio) fi) row)
  (logistic-regression-fit X
                           (on-vector (lr-case-y c) 'int32)
                           #:penalty penalty
                           #:C C
                           #:l1-ratio ratio
                           #:fit-intercept fi))

(define (row-name row)
  (match-define (list dt c (list penalty _ _) fi) row)
  (format "~a ~a ~a ~a" dt (lr-case-name c) penalty fi))

(test-twin "Python cuML's fit, decision and predict, bit for bit, and its probabilities within (k+3)u(1 + |log p|) (T1, T4)"
  (define kept (filter repeatable? rows))
  (for ([row (in-list kept)]
        [theirs (in-list (logistic-twin-results (map request kept)))])
    (match-define (list dt c _ _) row)
    (test-case (row-name row)
      (define X (on-matrix (lr-case-X c) dt))
      (define r (row-fit row X))
      (check-equal? (flat-parts r) (their-parts theirs dt) "fit")
      (check-equal? (flat (logistic-regression-decision-function r X))
                    (decoded dt (hash-ref theirs 'decision_bits))
                    "decision")
      (check-equal? (device-vector->list (logistic-regression-predict r X))
                    (hash-ref theirs 'predict)
                    "predict")
      (define k (classes-of c))
      (define proba
        (decoded (string->symbol (hash-ref theirs 'proba_dtype)) (hash-ref theirs 'proba_bits)))
      (define bound
        (if (and (= k 2) (eq? dt 'float32))
            (expt 2.0 -23)
            (unit-roundoff dt)))
      (define gap (proba-gap (flat (logistic-regression-predict-proba r X)) proba k))
      (check-true (<= gap bound) (format "proba gap ~a, ~a of the bound" gap (/ gap bound))))))

(test-twin "float32 multiclass OWL-QN fits, which repeat on neither side, within 1e-3 of Python cuML's (T1, F12)"
  (define kept (filter (lambda (row) (not (repeatable? row))) rows))
  (for ([row (in-list kept)]
        [theirs (in-list (logistic-twin-results (map (lambda (entry) (request entry #:op "cuml-fit"))
                                                     kept)))])
    (match-define (list dt c _ _) row)
    (test-case (row-name row)
      (match-define (list coef intercept _) (flat-parts (row-fit row (on-matrix (lr-case-X c) dt))))
      (match-define (list their-coef their-intercept _) (their-parts theirs dt))
      (define gap (relative-gap (append coef intercept) (append their-coef their-intercept)))
      (check-true (<= gap 1e-3) (format "gap ~a" gap)))))
