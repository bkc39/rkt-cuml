#lang racket/base

(require (only-in cuml adjusted-rand-score standard-scaler-fit-transform table->device-matrix)
         (only-in cuml/scribblings/guide/helpers counts-of renumbered rounded)
         (only-in cuml/scribblings/utils make-cuml-eval)
         (only-in data/order datum-order order-<?)
         (only-in datasets/core load-dataset-table)
         (only-in racket/file file->string)
         (only-in racket/function curry)
         (only-in racket/list append-map count take)
         (only-in racket/match match)
         (only-in racket/sandbox sandbox-namespace-specs)
         (only-in rackunit check-= check-equal?)
         (only-in raft device-vector->list)
         (only-in syntax/modread with-module-reading-parameterization)
         (only-in threading ~>)
         (only-in "scaler-data.rkt" transpose))

(provide boundary-margin
         check-scored
         cluster-columns
         evaluated-chapter
         example-blocks
         flattened-text
         host
         mismatches
         read-document
         scaled-matrix
         table-ari)

(define (read-document file)
  (call-with-input-file file
                        (lambda (in) (with-module-reading-parameterization (lambda () (read in))))))

(define (flattened-text path)
  (regexp-replace* #px"\\s+" (file->string path) " "))

(define (block-forms arguments)
  (match arguments
    [(list* (? keyword?) _ more) (block-forms more)]
    [forms forms]))

(define (example-blocks datum)
  (match datum
    [(list* 'examples arguments) (list (block-forms arguments))]
    [(? list?) (append-map example-blocks datum)]
    [_ '()]))

(define (evaluated-chapter name)
  (define chapter
    (parameterize ([sandbox-namespace-specs (append (sandbox-namespace-specs) '(cuml pict raft))])
      (make-cuml-eval)))
  (values chapter
          (~> (collection-file-path name "cuml" "scribblings" "guide" "examples")
              read-document
              example-blocks
              (map (curry map chapter) _))))

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

(define (cluster-columns counts)
  (sort (transpose counts) (order-<? datum-order)))

(define (host v n)
  (take (device-vector->list v) n))

(define (boundary-margin x places)
  (define scale (expt 10 places))
  (define scaled (* (inexact->exact x) scale))
  (/ (abs (- (- scaled (floor scaled)) 1/2)) scale))

(define (check-scored ids labels columns text)
  (check-equal? (cluster-columns (counts-of ids labels)) columns "the table's columns")
  (define ari (adjusted-rand-score ids labels))
  (check-= ari (table-ari columns) 1e-12 "the ARI is the table's")
  (check-equal? (rounded (table-ari columns)) text "the table's ARI, rounded")
  (check-equal? (rounded ari) text "the ARI, rounded"))
