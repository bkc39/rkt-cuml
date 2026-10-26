#lang racket/base

(require (only-in cuml standard-scaler-fit-transform table->device-matrix)
         (only-in datasets/core load-dataset-table)
         (only-in racket/list take)
         (only-in racket/match match*)
         (only-in raft device-vector->list)
         (only-in threading ~>)
         (only-in "scaler-data.rkt" transpose))

(provide boundary-margin
         cluster-columns
         host
         scaled-matrix
         table-ari)

(define (scaled-matrix name columns)
  (define-values (_fitted scaled)
    (~> (load-dataset-table name)
        (table->device-matrix #:columns columns)
        standard-scaler-fit-transform))
  scaled)

(define (pairs n)
  (/ (* n (sub1 n)) 2))

(define (pair-sum counts)
  (apply + (map pairs counts)))

(define (sums rows)
  (map (lambda (row) (apply + row)) rows))

(define (table-ari rows)
  (define within (pair-sum (apply append rows)))
  (define row-pairs (pair-sum (sums rows)))
  (define column-pairs (pair-sum (sums (transpose rows))))
  (define expected (/ (* row-pairs column-pairs) (pairs (apply + (sums rows)))))
  (define most (/ (+ row-pairs column-pairs) 2))
  (cond
    [(= most expected) 1]
    [else (/ (- within expected) (- most expected))]))

(define (before? a b)
  (match* (a b)
    [((cons x more) (cons y others)) (or (< x y) (and (= x y) (before? more others)))]
    [(_ _) (pair? b)]))

(define (cluster-columns counts)
  (sort (transpose counts) before?))

(define (host v n)
  (take (device-vector->list v) n))

(define (boundary-margin x places)
  (define scale (expt 10 places))
  (define scaled (* (inexact->exact x) scale))
  (/ (abs (- (- scaled (floor scaled)) 1/2)) scale))
