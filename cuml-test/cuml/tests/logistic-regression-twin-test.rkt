#lang racket/base

(require (only-in cuml/linear-model
                  logistic-regression-decision-function
                  logistic-regression-predict
                  logistic-regression-predict-proba)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-vector->list)
         (only-in "private/linear-data.rkt" by-dtype dtypes on-matrix unit-roundoff)
         (only-in "private/linear-twin-data.rkt" decoded)
         (only-in "private/logistic-data.rkt"
                  blob-case
                  fit-flat
                  fit-parts
                  flat-matrix
                  lr-case
                  lr-case-name
                  lr-case-w
                  lr-case-X
                  lr-case-y
                  lr-fit-n-iter
                  penalty-settings
                  relative-gap
                  setting-fit
                  setting-penalty
                  tight-max-iter
                  tight-tol)
         (only-in "private/logistic-twin-data.rkt" logistic-twin-results setting-twin their-fit)
         (only-in "private/python-env.rkt" test-twin))

(struct row (dt case setting fit-intercept))

(define cases
  (list (blob-case "binary" 150 4 2 21)
        (blob-case "three classes" 150 4 3 22)
        (blob-case "five classes" 200 4 5 23)))

(define rows
  (for*/list ([dt (in-list dtypes)]
              [c (in-list cases)]
              [s (in-list penalty-settings)]
              [fi (in-list '(#t #f))])
    (row dt c s fi)))

(define (request r #:op [op "cuml-apply"])
  (match-define (row dt c s fi) r)
  (setting-twin op c dt s #:fit-intercept fi))

(define summing-to-n
  (let* ([c (blob-case "binary weighted" 150 4 2 24 #:weighted? #t)]
         [w (lr-case-w c)]
         [scale (/ (length w) (apply + w))])
    (lr-case "binary weights summing to n"
             (lr-case-X c)
             (lr-case-y c)
             (map (lambda (v) (* v scale)) w))))

(define (weighted-gap-bound dt)
  (by-dtype dt 0 (min (* 64 1e-6) 3e-2)))

(define penalised (filter (lambda (s) (not (eq? (setting-penalty s) 'none))) penalty-settings))

(test-twin "weights summing to n fit as Python cuML's in float32 and near its optimum in float64"
  (define grid
    (for*/list ([dt (in-list dtypes)]
                [s (in-list penalised)])
      (cons dt s)))
  (define requests
    (for/list ([g (in-list grid)])
      (setting-twin "cuml-fit"
                    summing-to-n
                    (car g)
                    (cdr g)
                    #:tol (tight-tol (car g))
                    #:max-iter tight-max-iter)))
  (for ([g (in-list grid)]
        [theirs (in-list (logistic-twin-results requests))])
    (match-define (cons dt s) g)
    (test-case (format "~a ~a" dt (setting-penalty s))
      (define ours
        (fit-parts (setting-fit summing-to-n dt s #:tol (tight-tol dt) #:max-iter tight-max-iter)))
      (define gap (relative-gap (fit-flat ours) (fit-flat (their-fit theirs dt))))
      (check-true (<= gap (weighted-gap-bound dt)) (format "gap ~a" gap)))))

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

(define (repeatable? r)
  (match-define (row dt c s _) r)
  (not (and (eq? dt 'float32) (> (classes-of c) 2) (memq (setting-penalty s) '(l1 elasticnet)))))

(define (row-fit r)
  (match-define (row dt c s fi) r)
  (setting-fit c dt s #:fit-intercept fi))

(define (row-name r)
  (match-define (row dt c s fi) r)
  (format "~a ~a ~a ~a" dt (lr-case-name c) (setting-penalty s) fi))

(test-twin "the fit, decision and predict are Python cuML's bit for bit, and probabilities near its"
  (define kept (filter repeatable? rows))
  (for ([r (in-list kept)]
        [theirs (in-list (logistic-twin-results (map request kept)))])
    (match-define (row dt c _ _) r)
    (test-case (row-name r)
      (define X (on-matrix (lr-case-X c) dt))
      (define fitted (row-fit r))
      (check-equal? (fit-flat (fit-parts fitted)) (fit-flat (their-fit theirs dt)) "fit")
      (check-equal? (lr-fit-n-iter (fit-parts fitted)) (hash-ref theirs 'n_iter) "n-iter")
      (check-equal? (flat-matrix (logistic-regression-decision-function fitted X))
                    (decoded dt (hash-ref theirs 'decision_bits))
                    "decision")
      (check-equal? (device-vector->list (logistic-regression-predict fitted X))
                    (hash-ref theirs 'predict)
                    "predict")
      (define k (classes-of c))
      (define proba
        (decoded (string->symbol (hash-ref theirs 'proba_dtype)) (hash-ref theirs 'proba_bits)))
      (define bound
        (if (and (= k 2) (eq? dt 'float32))
            (expt 2.0 -23)
            (unit-roundoff dt)))
      (define gap (proba-gap (flat-matrix (logistic-regression-predict-proba fitted X)) proba k))
      (check-true (<= gap bound) (format "proba gap ~a, ~a of the bound" gap (/ gap bound))))))

(define owl-qn-gap-bound 1e-3)

(test-twin "float32 multiclass OWL-QN fits, which repeat on neither side, are near Python cuML's"
  (define kept (filter (lambda (r) (not (repeatable? r))) rows))
  (for ([r (in-list kept)]
        [theirs (in-list (logistic-twin-results (map (lambda (r) (request r #:op "cuml-fit")) kept)))])
    (match-define (row dt _ _ _) r)
    (test-case (row-name r)
      (define gap (relative-gap (fit-flat (fit-parts (row-fit r))) (fit-flat (their-fit theirs dt))))
      (check-true (<= gap owl-qn-gap-bound) (format "gap ~a" gap)))))
