#lang racket/base

(require (only-in cuml fit mbsgd-classifier mbsgd-regressor)
         (only-in cuml/linear-model
                  mbsgd-classifier-decision-function
                  mbsgd-classifier-fit
                  mbsgd-classifier-predict
                  mbsgd-regressor-fit
                  mbsgd-regressor-predict)
         (only-in rackunit check-exn test-case)
         (only-in raft list*->device-matrix)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/kmeans-data.rkt" blank-matrix)
         (only-in "private/linear-data.rkt" on-matrix on-vector)
         (only-in "private/refusal-rows.rkt" check-row row row-id))

(define grid '((1 0 2) (0 3 1) (4 1 0) (2 2 5) (3 1 1) (0 0 1)))
(define targets '(1 2 3 4 5 6))
(define labels '(0 1 0 1 1 0))
(define count-message "expected an exact integer from 1 to 2147483647, got ")

(define (settings-rows go X y)
  (list (row "penalty before alpha"
             "penalty: expected one of 'none 'l1 'l2 'elasticnet, got 'l3"
             (lambda () (go X y #:penalty 'l3 #:alpha 0)))
        (row "alpha before l1-ratio"
             "alpha: expected a positive finite real, got 0"
             (lambda () (go X y #:alpha 0 #:l1-ratio 2)))
        (row "l1-ratio before fit-intercept"
             "l1-ratio: expected a real in [0, 1], got 2"
             (lambda () (go X y #:l1-ratio 2 #:fit-intercept 'yes)))
        (row "fit-intercept before epochs"
             "fit-intercept: expected #t or #f, got 'yes"
             (lambda () (go X y #:fit-intercept 'yes #:epochs 0)))
        (row "epochs before tol"
             (string-append "epochs: " count-message "0")
             (lambda () (go X y #:epochs 0 #:tol -1)))
        (row "tol before shuffle"
             "tol: expected a nonnegative finite real, got -1"
             (lambda () (go X y #:tol -1 #:shuffle 1)))
        (row "shuffle before learning-rate"
             "shuffle: expected #t or #f, got 1"
             (lambda () (go X y #:shuffle 1 #:learning-rate 'optimal)))
        (row "learning-rate before eta0"
             "learning-rate: expected one of 'constant 'invscaling 'adaptive, got 'optimal"
             (lambda () (go X y #:learning-rate 'optimal #:eta0 0)))
        (row "eta0 before power-t"
             "eta0: expected a positive finite real, got 0"
             (lambda () (go X y #:eta0 0 #:power-t -1)))
        (row "power-t before batch-size"
             "power-t: expected a nonnegative finite real, got -1"
             (lambda () (go X y #:power-t -1 #:batch-size 0)))
        (row "batch-size before n-iter-no-change"
             (string-append "batch-size: " count-message "0")
             (lambda () (go X y #:batch-size 0 #:n-iter-no-change 0)))
        (row "n-iter-no-change before random-state"
             (string-append "n-iter-no-change: " count-message "0")
             (lambda () (go X y #:n-iter-no-change 0 #:random-state -1)))
        (row "random-state"
             "random-state: expected #f or an integer in [0, 4294967295], got -1"
             (lambda () (go X y #:random-state -1)))))

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
    (append (list (row "X kind before y"
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
            (shared-shim-rows mbsgd-regressor-fit on-vector)
            (list (row "a non-finite y"
                       "y: element 4: expected a finite value, got inf"
                       (lambda () (mbsgd-regressor-fit X (on-vector '(1 2 3 4 +inf.0 6))))))))
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
     (shared-shim-rows mbsgd-classifier-fit (lambda (ls) (on-vector ls 'int32)))
     (list (row "a non-integral device label"
                "y: expected integral class labels, got 1.5"
                (lambda () (mbsgd-classifier-fit X (on-vector '(0 1 0 1.5 1 0)))))
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
  (for ([apply
         (list mbsgd-regressor-predict mbsgd-classifier-predict mbsgd-classifier-decision-function)]
        [name (in-list '("mbsgd-regressor-predict" "mbsgd-classifier-predict"
                                                   "mbsgd-classifier-decision-function"))]
        [result (list fitted fitted-c fitted-c)]
        [predicate (in-list '("mbsgd-regressor-result?" "mbsgd-classifier-result?"
                                                        "mbsgd-classifier-result?"))])
    (test-case (format "~a result" name)
      (check-exn (lambda (e)
                   (and (exn:fail:contract? e)
                        (regexp-match? (regexp-quote (format "expected: ~a" predicate))
                                       (exn-message e))))
                 (lambda () (apply 'r X))))
    (for ([r (list (row "X kind before columns"
                        "X: expected a device-matrix, got 'X"
                        (lambda () (apply result 'X)))
                   (row "columns before dtype"
                        "X: expected 3 columns, got 2"
                        (lambda () (apply result (on-matrix '((1 2)) 'float32))))
                   (row "dtype"
                        "X: expected dtype float64 (the fit's), got float32"
                        (lambda () (apply result (on-matrix grid 'float32)))))])
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
