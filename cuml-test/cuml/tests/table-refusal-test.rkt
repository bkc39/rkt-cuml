#lang racket/base

(require (only-in cuml table->device-matrix)
         (only-in datasets/core dataset-missing load-dataset-table)
         (only-in rackunit check-equal?)
         (only-in raft/tests/private/collect drain-finalizers!)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in raft/tests/private/probe probe-pool-used test-pools)
         (only-in "private/refusal-rows.rkt" check-row row row-id)
         (only-in "private/table-data.rkt" synthetic-table))

(define refusal
  (make-keyword-procedure
   (lambda (kws kw-values id message table)
     (row id message (lambda () (keyword-apply table->device-matrix kws kw-values (list table)))))))

(define (rows)
  (define iris (load-dataset-table 'iris))
  (define airquality (load-dataset-table 'airquality))
  (define mixed (synthetic-table '("a" "b") (list (list 1 dataset-missing) (list "x" 2))))
  (list
   (refusal "T1" "table: expected a dataset table, got 5" 5)
   (refusal "T2" "columns: expected #f or a list of column names, got \"a\"" iris #:columns "a")
   (refusal "T2b"
            "columns: expected #f or a list of column names, got (sepal-length)"
            iris
            #:columns '(sepal-length))
   (refusal "T3" "columns: expected at least 1 column, got 0" iris #:columns '())
   (refusal "T4" "columns: expected distinct names, got \"a\" twice" iris #:columns '("a" "a"))
   (refusal "T5" "columns: expected a column of the table, got \"nope\"" iris #:columns '("nope"))
   (refusal "T6" "table: expected at least 1 column, got 0" (synthetic-table '() '()))
   (refusal "T7"
            "dtype: expected float32 or float64, got int32"
            iris
            #:columns '("sepal-length")
            #:dtype 'int32)
   (refusal "T8"
            "layout: expected row-major or col-major, got diagonal"
            iris
            #:columns '("sepal-length")
            #:layout 'diagonal)
   (refusal "T9"
            "resources: expected device resources, got #f"
            iris
            #:columns '("sepal-length")
            #:resources #f)
   (refusal "T10"
            "column \"ozone\": row 4: expected a number, got a missing value"
            airquality
            #:columns '("ozone"))
   (refusal "T11" "column \"species\": row 0: expected a real number, got \"Iris-setosa\"" iris)
   (refusal "T12"
            "column \"n\": row 1: expected a value float32 holds exactly, got 16777217"
            (synthetic-table '("n") (list (list (expt 2 24) (add1 (expt 2 24))))))
   (refusal "T12b"
            "column \"n\": row 1: expected a value float64 holds exactly, got 9007199254740993"
            (synthetic-table '("n") (list (list (expt 2 53) (add1 (expt 2 53)))))
            #:dtype 'float64)
   (refusal "T13"
            "column \"x\": row 0: expected a value within float32's range, got 1e+39"
            (synthetic-table '("x") '((1e39))))
   (refusal "T13b"
            "column \"x\": row 0: expected a value within float32's range, got 3.4028235677973366e+38"
            (synthetic-table '("x") '((3.4028235677973366e38))))
   (refusal "columns before dtype"
            "columns: expected a column of the table, got \"nope\""
            iris
            #:columns '("nope")
            #:dtype 'int32)
   (refusal "column-first scan, a then b"
            "column \"a\": row 1: expected a number, got a missing value"
            mixed
            #:columns '("a" "b"))
   (refusal "column-first scan, b then a"
            "column \"b\": row 0: expected a real number, got \"x\""
            mixed
            #:columns '("b" "a"))))

(test-gpu "L8: table->device-matrix refuses each bad argument with its message"
  (for ([r (in-list (rows))])
    (check-row "table->device-matrix" r)))

(test-pools "L9: a refusal allocates no device memory"
  (define device 0)
  (for ([r (in-list (rows))])
    (drain-finalizers!)
    (define before (probe-pool-used device))
    (check-row "table->device-matrix" r)
    (drain-finalizers!)
    (check-equal? (probe-pool-used device) before (row-id r))))
