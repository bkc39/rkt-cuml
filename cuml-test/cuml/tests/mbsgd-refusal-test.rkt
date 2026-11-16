#lang racket/base

(require (only-in cuml fit mbsgd-classifier mbsgd-regressor)
         (only-in cuml/linear-model
                  mbsgd-classifier-decision-function
                  mbsgd-classifier-fit
                  mbsgd-classifier-predict
                  mbsgd-regressor-fit
                  mbsgd-regressor-predict)
         (only-in racket/match match match-define)
         (only-in rackunit check-exn test-case)
         (only-in raft list*->device-matrix)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/elastic-net-data.rkt" apply-keywords)
         (only-in "private/kmeans-data.rkt" blank-matrix)
         (only-in "private/linear-data.rkt" on-matrix on-vector)
         (only-in "private/refusal-rows.rkt" check-row row row-id))

(define grid '((1 0 2) (0 3 1) (4 1 0) (2 2 5) (3 1 1) (0 0 1)))
(define targets '(1 2 3 4 5 6))
(define labels '(0 1 0 1 1 0))
(define count-message "expected an exact integer from 1 to 2147483647, got ")

(define settings-faults
  `((penalty l3 "expected one of 'none 'l1 'l2 'elasticnet, got 'l3")
    (alpha 0 "expected a positive finite real, got 0")
    (l1-ratio 2 "expected a real in [0, 1], got 2")
    (fit-intercept yes "expected #t or #f, got 'yes")
    (epochs 0 ,(string-append count-message "0"))
    (tol -1 "expected a nonnegative finite real, got -1")
    (shuffle 1 "expected #t or #f, got 1")
    (learning-rate optimal "expected one of 'constant 'invscaling 'adaptive, got 'optimal")
    (eta0 0 "expected a positive finite real, got 0")
    (power-t -1 "expected a nonnegative finite real, got -1")
    (batch-size 0 ,(string-append count-message "0"))
    (n-iter-no-change 0 ,(string-append count-message "0"))
    (random-state -1 "expected #f or an integer in [0, 4294967295], got -1")))

(define (keyword-of key)
  (string->keyword (symbol->string key)))

(define (settings-rows go X y)
  (for/list ([fault (in-list settings-faults)]
             [next (in-sequences (in-list (cdr settings-faults)) (in-value #f))])
    (match-define (list key bad message) fault)
    (define-values (id given)
      (match next
        [(list next-key next-bad _)
         (values (format "~a before ~a" key next-key)
                 (hash (keyword-of key) bad (keyword-of next-key) next-bad))]
        [#f (values (symbol->string key) (hash (keyword-of key) bad))]))
    (row id (format "~a: ~a" key message) (lambda () (apply-keywords go given (list X y))))))

(define (exact-value-rows go X y)
  (list
   (row "an exact alpha that rounds to 0.0"
        #rx"^alpha: expected a positive real representable as a nonzero flonum, got 1/10+[.][.][.]$"
        (lambda () (go X y #:alpha (expt 10 -400))))
   (row "an exact eta0 that rounds to 0.0"
        #rx"^eta0: expected a positive real representable as a nonzero flonum, got 1/10+[.][.][.]$"
        (lambda () (go X y #:eta0 (expt 10 -400))))
   (row "a negative exact tol that rounds to -0.0"
        #rx"^tol: expected a nonnegative finite real, got -1/10+[.][.][.]$"
        (lambda () (go X y #:tol (- (expt 10 -400)))))
   (row "an exact tol that rounds to +inf.0"
        #rx"^tol: expected a nonnegative finite real, got 10+[.][.][.]$"
        (lambda () (go X y #:tol (expt 10 400))))))

(define (shared-shim-rows go y-of)
  (list (row "X layout before y's length"
             "X: expected row-major, got col-major 6x3"
             (lambda ()
               (go (list*->device-matrix grid #:dtype 'float64 #:layout 'col-major) (y-of '(1 0)))))
        (row "one row before no column"
             "X: expected at least 2 rows, got 1"
             (lambda () (go (blank-matrix 1 0 'float64) (y-of '(1)))))
        (row "no column before y's length"
             "X: expected at least 1 column, got 0"
             (lambda () (go (blank-matrix 6 0 'float64) (y-of '(1 0)))))
        (row "y's length before a non-finite X"
             "y: expected 6 values (X's rows), got 5"
             (lambda () (go (on-matrix (cons '(+nan.0 0 0) (cdr grid))) (y-of '(0 1 0 1 1)))))
        (row "a non-finite X"
             "X: row 0, column 1: expected a finite value, got inf"
             (lambda () (go (on-matrix (cons '(1 +inf.0 0) (cdr grid))) (y-of labels))))))

(test-gpu "mbsgd-regressor-fit refuses each row, the earlier fault first"
  (define X (on-matrix grid))
  (define y (on-vector targets))
  (define rows
    (append
     (list (row "X kind before y"
                "X: expected a device-matrix, got '((1 2))"
                (lambda () (mbsgd-regressor-fit '((1 2)) 'y)))
           (row "y kind before the settings"
                "y: expected a device-vector, got '(1 2 3 4 5 6)"
                (lambda () (mbsgd-regressor-fit X targets #:loss 'hinge)))
           (row "y dtype before the settings"
                "y: expected dtype float64 (X's), got float32"
                (lambda () (mbsgd-regressor-fit X (on-vector targets 'float32) #:loss 'hinge)))
           (row "loss before penalty"
                "loss: expected 'squared-loss, got 'hinge"
                (lambda () (mbsgd-regressor-fit X y #:loss 'hinge #:penalty 'l3))))
     (settings-rows mbsgd-regressor-fit X y)
     (exact-value-rows mbsgd-regressor-fit X y)
     (shared-shim-rows mbsgd-regressor-fit on-vector)
     (list
      (row "a non-finite y"
           "y: element 4: expected a finite value, got inf"
           (lambda () (mbsgd-regressor-fit X (on-vector '(1 2 3 4 +inf.0 6)))))
      (row
       "an alpha beyond float32"
       "alpha: 1e+150 is not representable in float32"
       (lambda ()
         (mbsgd-regressor-fit (on-matrix grid 'float32) (on-vector targets 'float32) #:alpha 1e150)))
      (row
       "an eta0 below float32"
       "eta0: 1e-50 is not representable in float32"
       (lambda ()
         (mbsgd-regressor-fit (on-matrix grid 'float32) (on-vector targets 'float32) #:eta0 1e-50)))
      (row "a float64 y whose square overflows"
           "y is too large to check the fit; rescale y"
           (lambda () (mbsgd-regressor-fit X (on-vector (map (lambda (t) (* 1e155 t)) targets))))))))
  (for ([r (in-list rows)])
    (test-case (row-id r)
      (check-row "mbsgd-regressor-fit" r))))

(test-gpu "mbsgd-classifier-fit refuses each row, the earlier fault first"
  (define X (on-matrix grid))
  (define y (on-vector labels 'int32))
  (define rows
    (append
     (list (row "X kind before y"
                "X: expected a device-matrix, got '((1 2))"
                (lambda () (mbsgd-classifier-fit '((1 2)) 'y)))
           (row "y kind before the settings"
                (string-append "y: expected a device-vector of int32, int64, float32 or float64, "
                               "or a list of reals, strings or symbols, got 'y")
                (lambda () (mbsgd-classifier-fit X 'y #:loss 'optimal)))
           (row "loss before penalty"
                "loss: expected one of 'hinge 'log 'squared-loss, got 'optimal"
                (lambda () (mbsgd-classifier-fit X y #:loss 'optimal #:penalty 'l3)))
           (row "a host label before the settings' later faults"
                "y: expected integral class labels, got 1.5"
                (lambda () (mbsgd-classifier-fit X '(0 1 0 1.5 1 0))))
           (row "one host class"
                "y: expected exactly 2 classes, got 1"
                (lambda () (mbsgd-classifier-fit X '(a a a a a a))))
           (row "three host classes"
                "y: expected exactly 2 classes, got 3"
                (lambda () (mbsgd-classifier-fit X '(a b c a b c)))))
     (settings-rows mbsgd-classifier-fit X y)
     (exact-value-rows mbsgd-classifier-fit X y)
     (shared-shim-rows mbsgd-classifier-fit (lambda (ls) (on-vector ls 'int32)))
     (list (row "a non-integral device label"
                "y: expected integral class labels, got 1.5"
                (lambda () (mbsgd-classifier-fit X (on-vector '(0 1 0 1.5 1 0)))))
           (row "a power-t beyond float32"
                "power-t: 1e+40 is not representable in float32"
                (lambda () (mbsgd-classifier-fit (on-matrix grid 'float32) y #:power-t 1e40)))
           (row "one device class"
                "y: expected exactly 2 classes, got 1"
                (lambda () (mbsgd-classifier-fit X (on-vector '(4 4 4 4 4 4) 'int64))))
           (row "three device classes"
                "y: expected exactly 2 classes, got 3"
                (lambda () (mbsgd-classifier-fit X (on-vector '(0 1 2 0 1 2) 'int32)))))))
  (for ([r (in-list rows)])
    (test-case (row-id r)
      (check-row "mbsgd-classifier-fit" r))))

(test-gpu "predict and decision-function refuse a wrong result, X, its columns and its dtype"
  (define X (on-matrix grid))
  (define fitted (mbsgd-regressor-fit X (on-vector targets)))
  (define fitted-c (mbsgd-classifier-fit X labels))
  (for ([applier (in-list `((,mbsgd-regressor-predict "mbsgd-regressor-predict"
                                                      ,fitted
                                                      "mbsgd-regressor-result?")
                            (,mbsgd-classifier-predict "mbsgd-classifier-predict"
                                                       ,fitted-c
                                                       "mbsgd-classifier-result?")
                            (,mbsgd-classifier-decision-function "mbsgd-classifier-decision-function"
                                                                 ,fitted-c
                                                                 "mbsgd-classifier-result?")))])
    (match-define (list go name result predicate) applier)
    (test-case (format "~a result" name)
      (check-exn (lambda (e)
                   (and (exn:fail:contract? e)
                        (regexp-match? (regexp-quote (format "expected: ~a" predicate))
                                       (exn-message e))))
                 (lambda () (go 'r X))))
    (for ([r (list (row "X kind before columns"
                        "X: expected a device-matrix, got 'X"
                        (lambda () (go result 'X)))
                   (row "columns before dtype"
                        "X: expected 3 columns, got 2"
                        (lambda () (go result (on-matrix '((1 2)) 'float32))))
                   (row "dtype"
                        "X: expected dtype float64 (the fit's), got float32"
                        (lambda () (go result (on-matrix grid 'float32)))))])
      (test-case (format "~a ~a" name (row-id r))
        (check-row name r)))))

(test-gpu "the estimators refuse a sample weight"
  (define X (on-matrix grid))
  (for ([r (list
            (row "regressor"
                 "mbsgd-regressor takes no sample-weight"
                 (lambda ()
                   (fit (mbsgd-regressor) X (on-vector targets) #:sample-weight (on-vector targets))))
            (row "classifier"
                 "mbsgd-classifier takes no sample-weight"
                 (lambda () (fit (mbsgd-classifier) X labels #:sample-weight (on-vector targets)))))])
    (test-case (row-id r)
      (check-row "fit" r))))
