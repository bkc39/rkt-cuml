#lang racket/base

(require (only-in cuml adjusted-rand-score standard-scaler-fit-transform table->device-matrix)
         (only-in cuml/scribblings/guide/helpers counts-of renumbered rounded)
         (only-in datasets/core load-dataset-table)
         (only-in racket/file file->string)
         (only-in racket/list count take)
         (only-in racket/match == match match*)
         (only-in rackunit check-= check-equal?)
         (only-in raft device-vector->list)
         (only-in syntax/modread with-module-reading-parameterization)
         (only-in threading ~>)
         (only-in "scaler-data.rkt" transpose))

(provide boundary-margin
         chapter-clusterer
         check-scored
         cluster-columns
         flattened-text
         host
         mismatches
         read-document
         scaled-matrix
         table-ari)

(define-namespace-anchor anchor)

(define (read-document file)
  (call-with-input-file file
                        (lambda (in) (with-module-reading-parameterization (lambda () (read in))))))

(define (flattened-text path)
  (regexp-replace* #px"\\s+" (file->string path) " "))

(define (definition-body datum name)
  (match datum
    [(list 'define (== name) body) body]
    [(cons head tail) (or (definition-body head name) (definition-body tail name))]
    [_ #f]))

(define (chapter-clusterer)
  (define form
    (~> (collection-file-path "decomposition.scrbl" "cuml" "scribblings" "guide" "examples")
        read-document
        (definition-body 'clusterer)))
  (parameterize ([current-namespace (namespace-anchor->empty-namespace anchor)])
    (namespace-require 'racket/base)
    (namespace-require 'cuml)
    (eval form)))

(define (mismatches labels classes)
  (count (lambda (cluster class) (not (= cluster class))) labels (renumbered classes)))

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

(define (check-scored ids labels columns text)
  (check-equal? (cluster-columns (counts-of ids labels)) columns)
  (define ari (adjusted-rand-score ids labels))
  (check-= ari (table-ari columns) 1e-12)
  (check-equal? (rounded (table-ari columns)) text)
  (check-equal? (rounded ari) text))
