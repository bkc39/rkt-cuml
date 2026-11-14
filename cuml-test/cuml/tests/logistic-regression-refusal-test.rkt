#lang racket/base

(require (only-in cuml/linear-model
                  logistic-regression-decision-function
                  logistic-regression-fit
                  logistic-regression-predict
                  logistic-regression-result
                  logistic-regression-result-label-kind)
         (only-in racket/match match-define match-lambda struct*)
         (only-in rackunit check-equal? check-exn check-not-false test-case)
         (only-in raft device-vector->list device-vector? list*->device-matrix)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" on-matrix on-vector)
         (only-in "private/logistic-data.rkt" scaled-rows unequal-weights-message)
         (only-in "private/refusal-rows.rkt" check-row row row-id))

(define grid '((1 0 2) (0 3 1) (4 1 0) (2 2 5) (3 1 1) (0 0 1)))
(define labels '(0 1 0 1 1 0))
(define count-message "expected an exact integer from 1 to 2147483647, got ")

(define one-iteration-cutoff
  (regexp (string-append "^the solver stopped at first-order residual .* above the bound .* "
                         "after 1 iterations \\(max-iter 1, tol 1e-04\\); raise max-iter or tol$")))

(define (tiny-last rows factor)
  (map (match-lambda
         [(list a b c) (list a b (* factor c))])
       rows))

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
   (row "an exact C that rounds to 0.0"
        #rx"^C: expected a positive finite real, got 1/10+[.][.][.]$"
        (lambda () (go X y #:C (expt 10 -400))))
   (row "an exact C that rounds to +inf.0"
        #rx"^C: expected a positive finite real, got 10+[.][.][.]$"
        (lambda () (go X y #:C (expt 10 400))))
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
        "penalty-normalized: expected #t or #f, got 1"
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
   (row "two class-weight keys naming one class of a list"
        #rx"^class-weight: keys (1|1\\.0) and (1\\.0|1) both name class 1$"
        (lambda () (go X labels #:class-weight (hash 0 1 1 2 1.0 5))))
   (row "two class-weight keys naming one class of a device vector"
        #rx"^class-weight: keys (1|1\\.0) and (1\\.0|1) both name class 1$"
        (lambda () (go X y #:class-weight (hash 0 1 1 2 1.0 5))))
   (row "one class"
        "y: expected at least 2 classes, got 1"
        (lambda () (go X (on-vector '(4 4 4 4 4 4) 'int32))))
   (row "a class of zero weight"
        "y: class 1 has no weight"
        (lambda () (go X y #:sample-weight (on-vector '(1 0 1 0 0 1)))))
   (row "unequal weights with three classes"
        unequal-weights-message
        (lambda ()
          (go X (on-vector '(0 1 2 0 1 2) 'int32) #:sample-weight (on-vector '(1 2 1 1 1 1)))))
   (row "balanced weights with three unequal classes"
        unequal-weights-message
        (lambda () (go X (on-vector '(0 1 2 0 1 1) 'int32) #:class-weight 'balanced)))
   (row "uniform balanced weights with three unequal classes"
        unequal-weights-message
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
        (lambda () (go (on-matrix (scaled-rows grid (expt 2 -500))) y #:C 1e-300)))
   (row "a staged penalty outside float32"
        "C: the penalty at X's scale 2^66 is outside the float32 range"
        (lambda () (go (on-matrix (scaled-rows grid (expt 2 -66)) 'float32) y)))
   (row "a staged penalty beyond what float32's solver resolves"
        (regexp (string-append "^C: the penalty is too strong for the float32 solver to resolve at "
                               "X's scale 2\\^4[0-9]; raise C or standardise X$"))
        (lambda () (go (on-matrix (scaled-rows grid (expt 2 -40)) 'float32) y)))
   (row "a staged penalty beyond what float64's solver resolves"
        (regexp (string-append "^C: the penalty is too strong for the float64 solver to resolve at "
                               "X's scale 2\\^6[0-9]; raise C or standardise X$"))
        (lambda () (go (on-matrix (scaled-rows grid (expt 2 -60))) y)))
   (row "a subnormal float32 column with nothing staged"
        #rx"^X: column 2 has values only up to [^ ]+ in magnitude, subnormal in float32; scale it$"
        (lambda () (go (on-matrix (tiny-last grid 1e-40) 'float32) y)))
   (row "a subnormal float64 column with nothing staged"
        #rx"^X: column 2 has values only up to [^ ]+ in magnitude, subnormal in float64; scale it$"
        (lambda () (go (on-matrix (tiny-last grid 1e-310)) y)))
   (row "a coefficient beyond float32 once unstaged"
        "X: a coefficient exceeds the float32 range"
        (lambda ()
          (go (on-matrix (scaled-rows '((1 0) (0 1) (2 1) (3 0) (5 1) (4 0)) (expt 2 -124)) 'float32)
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

(define (relabelled fitted kind)
  (match-define (struct* logistic-regression-result
                         ([coef coef] [intercept intercept] [classes classes] [n-iter n-iter]))
    fitted)
  (logistic-regression-result coef intercept classes kind n-iter))

(define (host v)
  (if (device-vector? v)
      (device-vector->list v)
      v))

(test-gpu "a result built by hand takes only a fitted result's label kind"
  (define X (on-matrix grid))
  (for ([y (list (on-vector '(-3 2 7 -3 2 7) 'int32) '("x" "y" "z" "x" "y" "z"))])
    (define fitted (logistic-regression-fit X y))
    (for ([kind (list 'int32 'list #f)])
      (test-case (format "label-kind ~e" kind)
        (check-row "logistic-regression-result"
                   (row (format "~e" kind)
                        (format "label-kind: expected the label kind of a fitted result, got ~e" kind)
                        (lambda () (relabelled fitted kind))))))
    (test-case "a copied label kind still answers in labels"
      (define copied (relabelled fitted (logistic-regression-result-label-kind fitted)))
      (define labels (host (logistic-regression-predict copied X)))
      (check-equal? labels (host (logistic-regression-predict fitted X)))
      (check-not-false (andmap (lambda (l) (member l (host y))) labels)))))
