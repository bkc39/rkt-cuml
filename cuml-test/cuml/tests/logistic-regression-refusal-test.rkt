#lang racket/base

(require (only-in cuml/linear-model
                  logistic-regression-decision-function
                  logistic-regression-fit
                  logistic-regression-predict)
         (only-in racket/match match-define)
         (only-in rackunit check-exn test-case)
         (only-in raft list*->device-matrix)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" on-matrix on-vector)
         (only-in "private/refusal-rows.rkt" check-row row row-id))

(define grid '((1 0 2) (0 3 1) (4 1 0) (2 2 5) (3 1 1) (0 0 1)))
(define labels '(0 1 0 1 1 0))
(define count-message "expected an exact integer from 1 to 2147483647, got ")

(define one-iteration-cutoff
  (regexp (string-append "^the solver stopped at first-order residual .* above the bound .* "
                         "after 1 iterations \\(max-iter 1, tol 1e-04\\); raise max-iter or tol$")))

(define (scaled rows factor)
  (for/list ([r (in-list rows)])
    (map (lambda (v) (* v factor)) r)))

(define (overflowing rows)
  (define huge (* 3/2 (expt 2 1023)))
  (for/list ([r (in-list rows)]
             [i (in-naturals)])
    (cons (if (even? i)
              huge
              (* 3/4 huge))
          (cdr r))))

(define (racket-rows go)
  (define X (on-matrix grid))
  (define y (on-vector labels 'int32))
  (list
   (row "X kind" "X: expected a device-matrix, got '((1 2))" (lambda () (go '((1 2)) y)))
   (row "y kind before the hyperparameters"
        (string-append "y: expected a device-vector of int32, int64, float32 or float64, "
                       "or a list of reals, strings or symbols, got 'y")
        (lambda () (go X 'y #:penalty 'x)))
   (row "y of mixed kinds"
        "y: expected labels of one kind (reals, strings or symbols), got 0 and \"a\""
        (lambda () (go X '(0 1 0 "a" 1 0))))
   (row "penalty"
        "penalty: expected 'l1, 'l2, 'elasticnet or 'none, got 'l3"
        (lambda () (go X y #:penalty 'l3 #:C 0)))
   (row "C zero" "C: expected a positive finite real, got 0" (lambda () (go X y #:C 0)))
   (row "C infinite" "C: expected a positive finite real, got +inf.0" (lambda () (go X y #:C +inf.0)))
   (row "C nan" "C: expected a positive finite real, got +nan.0" (lambda () (go X y #:C +nan.0)))
   (row "elastic net without l1-ratio"
        "l1-ratio: expected a real from 0 to 1 with 'elasticnet, got #f"
        (lambda () (go X y #:penalty 'elasticnet)))
   (row "l1-ratio above 1"
        "l1-ratio: expected a real from 0 to 1 with 'elasticnet, got 1.5"
        (lambda () (go X y #:penalty 'elasticnet #:l1-ratio 1.5)))
   (row "l1-ratio with l2"
        "l1-ratio: given with penalty 'l2; expected #f"
        (lambda () (go X y #:l1-ratio 0.5)))
   (row "fit-intercept"
        "fit-intercept: expected #t or #f, got 'yes"
        (lambda () (go X y #:fit-intercept 'yes)))
   (row "max-iter 0" (string-append "max-iter: " count-message "0") (lambda () (go X y #:max-iter 0)))
   (row "linesearch-max-iter 0"
        (string-append "linesearch-max-iter: " count-message "0")
        (lambda () (go X y #:linesearch-max-iter 0)))
   (row "lbfgs-memory 0, which segfaults upstream"
        (string-append "lbfgs-memory: " count-message "0")
        (lambda () (go X y #:lbfgs-memory 0)))
   (row "negative tol"
        "tol: expected a nonnegative finite real, got -1"
        (lambda () (go X y #:tol -1)))
   (row "nan tol"
        "tol: expected a nonnegative finite real, got +nan.0"
        (lambda () (go X y #:tol +nan.0)))
   (row "penalty-normalized"
        "penalty-normalized: expected a boolean, got 1"
        (lambda () (go X y #:penalty-normalized 1)))
   (row
    "class-weight kind"
    "class-weight: expected #f, 'balanced or a hash from labels to positive finite reals, got 'even"
    (lambda () (go X y #:class-weight 'even)))
   (row
    "negative class weight"
    "class-weight: expected #f, 'balanced or a hash from labels to positive finite reals, got '#hash((0 . -1))"
    (lambda () (go X y #:class-weight (hash 0 -1))))
   (row "sample-weight kind"
        "sample-weight: expected a device-vector or #f, got '(1 1)"
        (lambda () (go X y #:sample-weight '(1 1))))))

(define (shim-rows go)
  (define X (on-matrix grid))
  (define y (on-vector labels 'int32))
  (list
   (row "X layout"
        "X: expected row-major, got col-major 6x3"
        (lambda () (go (list*->device-matrix grid #:dtype 'float64 #:layout 'col-major) y)))
   (row "y length" "y: expected 6 elements, got 5" (lambda () (go X (on-vector '(0 1 0 1 1) 'int32))))
   (row "non-integral label"
        "y: expected integral class labels, got 1.5"
        (lambda () (go X (on-vector '(0 1 0 1.5 1 0)))))
   (row "an infinite label"
        "y: expected integral class labels, got inf"
        (lambda () (go X (on-vector '(0 1 0 +inf.0 1 0)))))
   (row "a negative infinite label"
        "y: expected integral class labels, got -inf"
        (lambda () (go X (on-vector '(0 1 0 -inf.0 1 0)))))
   (row "a nan label"
        "y: expected integral class labels, got nan"
        (lambda () (go X (on-vector '(0 1 0 +nan.0 1 0)))))
   (row "non-integral list label"
        "y: expected integral class labels, got 1.5"
        (lambda () (go X '(0 1 0 1.5 1 0))))
   (row "non-finite X"
        "X: row 3, column 1: expected a finite value, got nan"
        (lambda () (go (on-matrix '((1 0 2) (0 3 1) (4 1 0) (2 +nan.0 5) (3 1 1) (0 0 1))) y)))
   (row "negative weight"
        "sample-weight: element 1: expected at least 0, got -1"
        (lambda () (go X y #:sample-weight (on-vector '(1 -1 1 1 1 1)))))
   (row "one row"
        "X: expected at least 2 rows, got 1"
        (lambda () (go (on-matrix '((1 2 3))) (on-vector '(0) 'int32))))
   (row "a key that is not a class with a class missing"
        "class-weight: classes (0) are not in class-weight"
        (lambda () (go X y #:class-weight (hash 1 2.0 7 3.0))))
   (row "a list key that is not a class with a class missing"
        "class-weight: classes (\"a\") are not in class-weight"
        (lambda () (go X '("a" "b" "a" "b" "b" "a") #:class-weight (hash "b" 2.0 "c" 3.0))))
   (row "one class"
        "y: expected at least 2 classes, got 1"
        (lambda () (go X (on-vector '(4 4 4 4 4 4) 'int32))))
   (row "a class of zero weight"
        "y: class 1 has no weight"
        (lambda () (go X y #:sample-weight (on-vector '(1 0 1 0 0 1)))))
   (row "unequal weights with three classes"
        (string-append "sample-weight: weights with 3 or more classes must be equal on every row "
                       "(cuML's softmax ignores them); got unequal weights")
        (lambda ()
          (go X (on-vector '(0 1 2 0 1 2) 'int32) #:sample-weight (on-vector '(1 2 1 1 1 1)))))
   (row "balanced weights with three unequal classes"
        (string-append "sample-weight: weights with 3 or more classes must be equal on every row "
                       "(cuML's softmax ignores them); got unequal weights")
        (lambda () (go X (on-vector '(0 1 2 0 1 1) 'int32) #:class-weight 'balanced)))
   (row "uniform balanced weights with three unequal classes"
        (string-append "sample-weight: weights with 3 or more classes must be equal on every row "
                       "(cuML's softmax ignores them); got unequal weights")
        (lambda ()
          (go X
              (on-vector '(0 1 2 0 1 1) 'int32)
              #:sample-weight (on-vector '(0.1 0.1 0.1 0.1 0.1 0.1))
              #:class-weight 'balanced)))
   (row "a column whose spread overflows"
        (string-append "X: column 0 has values up to 1.348269851146737e+308 in magnitude; "
                       "its spread is outside the float64 range")
        (lambda () (go (on-matrix (overflowing grid)) y)))
   (row "a staged penalty outside float64"
        "C: the penalty at X's scale 2^500 is outside the float64 range"
        (lambda () (go (on-matrix (scaled grid (expt 2 -500))) y #:C 1e-300)))
   (row "a staged penalty outside float32"
        "C: the penalty at X's scale 2^66 is outside the float32 range"
        (lambda () (go (on-matrix (scaled grid (expt 2 -66)) 'float32) y)))
   (row "a coefficient beyond float32 once unstaged"
        "X: a coefficient exceeds the float32 range"
        (lambda ()
          (go (on-matrix (scaled '((1 0) (0 1) (2 1) (3 0) (5 1) (4 0)) (expt 2 -124)) 'float32)
              (on-vector '(0 0 0 1 1 1) 'int32)
              #:penalty 'none)))
   (row "an iteration cutoff" one-iteration-cutoff (lambda () (go X y #:max-iter 1)))))

(test-gpu "logistic-regression-fit refuses each row, the earlier fault first"
  (for ([r (in-list (append (racket-rows logistic-regression-fit)
                            (shim-rows logistic-regression-fit)))])
    (test-case (row-id r)
      (check-row "logistic-regression-fit" r))))

(struct applied-row (function row))

(define (predict-row id message call)
  (applied-row "logistic-regression-predict" (row id message call)))

(test-gpu "the apply functions refuse a wrong result, X, its columns and its dtype"
  (define fitted (logistic-regression-fit (on-matrix grid) labels))
  (test-case "result"
    (check-exn (lambda (e)
                 (and (exn:fail:contract? e)
                      (regexp-match? (regexp-quote "expected: logistic-regression-result?")
                                     (exn-message e))))
               (lambda () (logistic-regression-predict 'r (on-matrix grid)))))
  (for ([r (in-list
            (list
             (predict-row "X kind"
                          "X: expected a device-matrix, got 'X"
                          (lambda () (logistic-regression-predict fitted 'X)))
             (applied-row
              "logistic-regression-decision-function"
              (row "columns"
                   "X: expected 3 columns, got 2"
                   (lambda () (logistic-regression-decision-function fitted (on-matrix '((1 2)))))))
             (predict-row "X layout"
                          "X: expected row-major, got col-major 6x3"
                          (lambda ()
                            (logistic-regression-predict
                             fitted
                             (list*->device-matrix grid #:dtype 'float64 #:layout 'col-major))))
             (predict-row "dtype"
                          "X: expected dtype float64 (the fit's), got float32"
                          (lambda () (logistic-regression-predict fitted (on-matrix grid 'float32))))
             (predict-row "non-finite X"
                          "X: row 0, column 0: expected a finite value, got inf"
                          (lambda ()
                            (logistic-regression-predict fitted (on-matrix '((+inf.0 0 0))))))))])
    (match-define (applied-row function checked) r)
    (test-case (row-id checked)
      (check-row function checked))))
