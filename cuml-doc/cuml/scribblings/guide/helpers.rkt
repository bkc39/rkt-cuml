#lang racket/base

(require cuml
         datasets/core
         racket/function
         racket/list
         racket/match
         racket/math
         raft
         threading)

(provide agreement
         crosstab
         dataset-column
         dataset-columns
         dataset-matrix
         features
         labels-of
         rows
         sizes
         standardized
         standardized-matrix)

(define (crosstab labels classes)
  (define names (remove-duplicates classes))
  (define (counts cluster)
    (for/list ([name (in-list names)])
      (count (lambda (sample) (equal? (cdr sample) name)) cluster)))
  (~> (map cons labels classes)
      (filter (lambda (sample) (>= (car sample) 0)) _)
      (group-by car _)
      (map counts _)
      (sort < #:key (lambda (row) (index-of row (apply max row))))))

(define (agreement table)
  (apply + (map (curry apply max) table)))

(define (sizes labels)
  (~> (remove* '(-1) labels) (group-by values _) (map length _) (sort >)))

(define (standardized columns)
  (define (standardize column)
    (define n (length column))
    (define mean (/ (apply + column) n))
    (define sd (sqrt (/ (for/sum ([v (in-list column)]) (sqr (- v mean))) n)))
    (define scale (if (zero? sd) 1 sd))
    (map (lambda (v) (exact->inexact (/ (- v mean) scale))) column))
  (match (memf (negate real?) (append* columns))
    [(cons value _) (raise-argument-error 'standardized "real?" value)]
    [#f (map standardize columns)]))

(define (rows columns)
  (apply map list columns))

(define (dataset-columns name columns)
  (~> (load-dataset-table name #:columns columns)
      dataset-table-columns
      vector->list
      (map vector->list _)))

(define (dataset-column name column)
  (match-define (list column-values) (dataset-columns name (list column)))
  column-values)

(define (features name . excluded)
  (~> (load-dataset-table name) dataset-table-names (remove* excluded _)))

(define (dataset-matrix name columns #:dtype [dtype 'float32])
  (~> (dataset-columns name columns) rows (list*->device-matrix #:dtype dtype)))

(define (standardized-matrix name columns)
  (~> (dataset-columns name columns) standardized rows (list*->device-matrix #:dtype 'float32)))

(define (labels-of estimator samples)
  (device-vector->list (fit-predict estimator samples)))
