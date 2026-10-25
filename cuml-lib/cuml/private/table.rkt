#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in datasets/core
                  dataset-missing?
                  dataset-table-column
                  dataset-table-names
                  dataset-table-row-count
                  dataset-table?)
         (only-in racket/flonum flsingle)
         (only-in racket/list check-duplicates)
         (only-in racket/match match)
         (only-in racket/math infinite?)
         (only-in raft current-device-resources device-matrix device-resources? list*->device-matrix))

(provide table->device-matrix
         table->device-matrix/who)

(define element-types '(float32 float64))

(define orders '(row-major col-major))

(define (rounded element-type x)
  (match element-type
    ['float32 (flsingle x)]
    ['float64 x]))

(define (held-exactly? element-type v)
  (= (rounded element-type (real->double-flonum v)) v))

(define (beyond-range? element-type v)
  (and (not (infinite? v)) (infinite? (rounded element-type v))))

(define (cell-problem element-type v)
  (cond
    [(dataset-missing? v) "expected a number, got a missing value"]
    [(not (real? v)) (format "expected a real number, got ~s" v)]
    [(and (exact? v) (not (held-exactly? element-type v)))
     (format "expected a value ~a holds exactly, got ~s" element-type v)]
    [(and (inexact? v) (beyond-range? element-type v))
     (format "expected a value within ~a's range, got ~s" element-type v)]
    [else #f]))

(define (column-problem element-type name column)
  (for/or ([v (in-vector column)]
           [row (in-naturals)])
    (define problem (cell-problem element-type v))
    (and problem (format "column ~s: row ~a: ~a" name row problem))))

(define (selection who table columns)
  (define names (and (dataset-table? table) (dataset-table-names table)))
  (define names? (and (list? columns) (andmap string? columns)))
  (define duplicate (and names? (check-duplicates columns)))
  (define unknown (and names names? (findf (lambda (name) (not (member name names))) columns)))
  (cond
    [(not names) (logic-error "~a: table: expected a dataset table, got ~s" who table)]
    [(not (or (not columns) names?))
     (logic-error "~a: columns: expected #f or a list of column names, got ~s" who columns)]
    [(null? columns) (logic-error "~a: columns: expected at least 1 column, got 0" who)]
    [duplicate (logic-error "~a: columns: expected distinct names, got ~s twice" who duplicate)]
    [unknown (logic-error "~a: columns: expected a column of the table, got ~s" who unknown)]
    [(and (not columns) (null? names))
     (logic-error "~a: table: expected at least 1 column, got 0" who)]
    [else (or columns names)]))

(define (check-settings who element-type order resources)
  (cond
    [(not (memq element-type element-types))
     (logic-error "~a: dtype: expected float32 or float64, got ~s" who element-type)]
    [(not (memq order orders))
     (logic-error "~a: layout: expected row-major or col-major, got ~s" who order)]
    [(not (device-resources? resources))
     (logic-error "~a: resources: expected device resources, got ~s" who resources)]
    [else (void)]))

(define/who (table->device-matrix who
                                  table
                                  #:columns [columns #f]
                                  #:dtype [element-type 'float32]
                                  #:layout [order 'row-major]
                                  #:resources [resources (current-device-resources)])
  (define selected (selection who table columns))
  (check-settings who element-type order resources)
  (define vectors (map (lambda (name) (dataset-table-column table name)) selected))
  (define problem
    (for/or ([name (in-list selected)]
             [column (in-list vectors)])
      (column-problem element-type name column)))
  (define n (dataset-table-row-count table))
  (cond
    [problem (logic-error "~a: ~a" who problem)]
    [(zero? n)
     (device-matrix 0 (length selected) #:dtype element-type #:layout order #:resources resources)]
    [else
     (list*->device-matrix (for/list ([row (in-range n)])
                             (for/list ([column (in-list vectors)])
                               (vector-ref column row)))
                           #:dtype element-type
                           #:layout order
                           #:resources resources)]))
