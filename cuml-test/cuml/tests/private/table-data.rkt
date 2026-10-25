#lang racket/base

(require (only-in datasets/core
                  dataset-names
                  dataset-table-columns
                  dataset-table-names
                  dataset-table-row-count
                  load-dataset-table)
         (only-in racket/list filter-map index-of)
         (only-in racket/match match))

(provide edge-table
         edge-table32
         edge-values
         edge-values32
         host-rows
         numeric-selections
         (struct-out selection)
         synthetic-table)

(struct selection (name table columns))

(define-values (table-type skipped?) (struct-info (load-dataset-table 'anscombe #:columns '("x1"))))

(define reflected-table (struct-type-make-constructor table-type))

(define (synthetic-table names columns)
  (reflected-table names
                   (apply vector-immutable (map (lambda (c) (apply vector-immutable c)) columns))
                   (match columns
                     ['() 0]
                     [(cons column _) (length column)])
                   (hash "name" "synthetic")))

(define (all-real? table name)
  (for/and ([v (in-vector (vector-ref (dataset-table-columns table)
                                      (index-of (dataset-table-names table) name)))])
    (real? v)))

(define (numeric-selections)
  (filter-map (lambda (name)
                (define table (load-dataset-table name))
                (match (filter (lambda (column) (all-real? table column)) (dataset-table-names table))
                  ['() #f]
                  [columns (selection name table columns)]))
              (dataset-names)))

(define (host-rows table columns [convert values])
  (define vectors
    (for/list ([name (in-list columns)])
      (vector-ref (dataset-table-columns table) (index-of (dataset-table-names table) name))))
  (for/list ([i (in-range (dataset-table-row-count table))])
    (for/list ([column (in-list vectors)])
      (convert (vector-ref column i)))))

(define edge-values32
  (list -0.0 +nan.0 +inf.0 -inf.0 5e-324 0 (expt 2 24) (- (expt 2 24)) 0.5 1e-45 1e-46))

(define edge-values (append edge-values32 (list 1.7976931348623157e308 (expt 2 53) (- (expt 2 53)))))

(define (edge-of xs)
  (synthetic-table '("x" "i") (list xs (build-list (length xs) values))))

(define edge-table (edge-of edge-values))

(define edge-table32 (edge-of edge-values32))
