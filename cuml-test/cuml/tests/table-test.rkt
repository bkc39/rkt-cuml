#lang racket/base

(require (only-in cuml exn:fail:cuml? table->device-matrix)
         (only-in datasets/core dataset-table-names dataset-table-row-count load-dataset-table)
         (only-in racket/flonum flsingle)
         (only-in racket/list drop range take)
         (only-in racket/match match match-define)
         (only-in rackunit check-equal?)
         (only-in raft
                  contiguous?
                  current-device-resources
                  device-matrix->list*
                  device-resources
                  dtype
                  exn:fail:raft?
                  layout
                  list*->device-matrix
                  shape
                  with-device-resources)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" raised)
         (only-in "private/program.rkt" run-program)
         (only-in "private/table-data.rkt"
                  edge-table
                  edge-table32
                  host-rows
                  numeric-selections
                  selection
                  synthetic-table)
         (only-in threading ~>))

(define dtypes '(float32 float64))

(define (float32-of v)
  (flsingle (real->double-flonum v)))

(define (converted-rows t columns element-type)
  (host-rows t
             columns
             (match element-type
               ['float64 real->double-flonum]
               ['float32 float32-of])))

(define (matrix-of s element-type)
  (match-define (selection _ t columns) s)
  (table->device-matrix t #:columns columns #:dtype element-type))

(test-gpu "L1a: the shape is the row count by the selection, and the dtype is the one given"
  (for* ([s (in-list (numeric-selections))]
         [element-type (in-list dtypes)])
    (match-define (selection name t columns) s)
    (define M (matrix-of s element-type))
    (check-equal? (shape M) (list (dataset-table-row-count t) (length columns)) (format "~a" name))
    (check-equal? (dtype M) element-type)))

(test-gpu "L1b: a one-column table, and a zero-row table keeps its column count"
  (check-equal? (~> (synthetic-table '("a") '((1 2 3))) table->device-matrix shape) '(3 1))
  (for ([element-type (in-list dtypes)])
    (define M (table->device-matrix (synthetic-table '("a" "b") '(() ())) #:dtype element-type))
    (check-equal? (list (shape M) (dtype M)) (list '(0 2) element-type))))

(test-gpu "L2: float64 round-trips every numeric selection and the edge values exactly"
  (for ([s (in-list (numeric-selections))])
    (match-define (selection name t columns) s)
    (check-equal? (device-matrix->list* (matrix-of s 'float64))
                  (converted-rows t columns 'float64)
                  (format "~a" name)))
  (check-equal? (device-matrix->list* (table->device-matrix edge-table #:dtype 'float64))
                (converted-rows edge-table '("x" "i") 'float64)))

(test-gpu "L3a: float32 is flsingle of the double, for every numeric selection and the edge values"
  (for ([s (in-list (numeric-selections))])
    (match-define (selection name t columns) s)
    (check-equal? (device-matrix->list* (matrix-of s 'float32))
                  (converted-rows t columns 'float32)
                  (format "~a" name)))
  (check-equal? (device-matrix->list* (table->device-matrix edge-table32))
                (converted-rows edge-table32 '("x" "i") 'float32)))

(define (round-trip column)
  (~> (synthetic-table '("v") (list column)) table->device-matrix device-matrix->list* (map car _)))

(test-gpu "L3b: values float32 represents come back equal, and underflow is pinned"
  (define representable
    (list 0
          1
          -1
          (expt 2 24)
          (- (expt 2 24))
          (sub1 (expt 2 24))
          (expt 2 127)
          1/2
          0.5
          (flsingle 5.1)
          (flsingle 0.1)))
  (check-equal? (round-trip representable) (map real->double-flonum representable))
  (check-equal? (round-trip '(1e-46 1e-45)) '(0.0 1.401298464324817e-45)))

(test-gpu "L3c: a double below the halfway point rounds to float32's maximum"
  (check-equal? (round-trip '(3.4028235e38)) '(3.4028234663852886e+38)))

(define (rotations columns)
  (for/list ([k (in-range (length columns))])
    (append (drop columns k) (take columns k))))

(test-gpu "L4a: a permuted selection permutes the columns of every row"
  (for ([s (in-list (numeric-selections))])
    (match-define (selection name t columns) s)
    (define by-name
      (for/hash ([column (in-list columns)]
                 [values-of
                  (in-list (~> (matrix-of s 'float64) device-matrix->list* (apply map list _)))])
        (values column values-of)))
    (for ([permuted (in-list (cons (reverse columns) (rotations columns)))])
      (define expected (apply map list (map (lambda (column) (hash-ref by-name column)) permuted)))
      (check-equal?
       (device-matrix->list* (table->device-matrix t #:columns permuted #:dtype 'float64))
       expected
       (format "~a ~a" name permuted)))))

(test-gpu "L4b: #:columns #f is every column in the table's order"
  (define t (load-dataset-table 'anscombe))
  (for ([element-type (in-list dtypes)])
    (check-equal? (device-matrix->list* (table->device-matrix t #:dtype element-type))
                  (device-matrix->list*
                   (table->device-matrix t #:columns (dataset-table-names t) #:dtype element-type)))))

(test-gpu "L5: the result is list*->device-matrix of the table's rows"
  (for* ([s (in-list (numeric-selections))]
         [element-type (in-list dtypes)])
    (match-define (selection name t columns) s)
    (define M (matrix-of s element-type))
    (define L (list*->device-matrix (host-rows t columns) #:dtype element-type))
    (check-equal? (list (shape M) (dtype M) (device-matrix->list* M))
                  (list (shape L) (dtype L) (device-matrix->list* L))
                  (format "~a ~a" name element-type))))

(test-gpu "L6: col-major holds the same rows, and the default is a contiguous row-major matrix"
  (define t (load-dataset-table 'anscombe))
  (define row-major (table->device-matrix t))
  (define col-major (table->device-matrix t #:layout 'col-major))
  (check-equal? (list (layout row-major) (contiguous? row-major) (layout col-major))
                '(row-major #t col-major))
  (check-equal? (device-matrix->list* col-major) (device-matrix->list* row-major)))

(define (released-resources)
  (with-device-resources ([r (device-resources)]) r))

(define (released-default)
  (with-device-resources ([r (current-device-resources)]) r))

(define (released-message who)
  (format "~a: the device resources on device 0 were released" who))

(define (check-raft-only who thunk)
  (define e (raised thunk))
  (check-equal? (list (exn:fail:raft? e) (exn:fail:cuml? e) (and e (exn-message e)))
                (list #t #f (released-message who))))

(test-gpu "L7: released resources raise raft's error, and the default is read at each call"
  (define t (load-dataset-table 'anscombe))
  (check-raft-only "list*->device-matrix"
                   (lambda () (table->device-matrix t #:resources (released-resources))))
  (check-raft-only
   "device-matrix"
   (lambda () (table->device-matrix (synthetic-table '("a") '(())) #:resources (released-resources))))
  (check-raft-only "list*->device-matrix"
                   (lambda () (table->device-matrix t #:resources (released-default))))
  (check-equal? (~> t table->device-matrix shape) '(11 8)))

(test-gpu "L10: cuml/table loads no cuML family or cuML native library"
  (check-equal?
   (run-program `(module m racket/base
                   (require cuml/table)
                   (exit (if (ormap (lambda (m) (module-declared? m #f))
                                    '(cuml/cluster/private/native cuml/core/private/native))
                             1
                             0))))
   '(0 "" "")))

(test-gpu "L11: the table is unchanged by a conversion"
  (for ([s (in-list (numeric-selections))])
    (match-define (selection name t _) s)
    (matrix-of s 'float32)
    (check-equal? t (load-dataset-table name) (format "~a" name))))

(test-gpu "L1c: a matrix as wide as the table"
  (check-equal? (~> (range 40)
                    (map number->string _)
                    (synthetic-table (build-list 40 (lambda (i) (list i (- i)))))
                    table->device-matrix
                    shape)
                '(2 40)))
