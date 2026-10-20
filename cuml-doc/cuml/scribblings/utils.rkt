#lang racket/base

(require (for-label cuml
                    raft
                    raft/compat
                    racket/base
                    racket/list
                    racket/math)
         (only-in racket/sandbox
                  sandbox-error-output
                  sandbox-eval-limits
                  sandbox-memory-limit
                  sandbox-output
                  sandbox-path-permissions
                  sandbox-security-guard)
         scribble/example
         scribble/manual
         syntax/parse/define)

(provide (all-from-out scribble/example)
         (all-from-out scribble/manual)
         (for-label (all-from-out cuml raft raft/compat racket/base racket/list racket/math))
         make-cuml-eval
         page-helpers-block)

(define-syntax-parse-rule (define-page-helpers forms:id block:id form:expr ...)
  (begin
    (define forms '(form ...))
    (define block (racketblock form ...))))

(define-page-helpers page-helpers page-helpers-block
  (define (crosstab labels classes)
    (define names (remove-duplicates classes))
    (~> (for/list ([cluster (in-list (group-by car (map cons labels classes)))])
          (for/list ([name (in-list names)])
            (count (lambda (sample) (equal? (cdr sample) name)) cluster)))
        (sort < #:key (lambda (row) (index-of row (apply max row))))))
  (define (agreement table)
    (for/sum ([row (in-list table)]) (apply max row)))
  (define (sizes labels)
    (sort (map length (group-by values labels)) >))
  (define (standardized-rows table)
    (~> (for/list ([column (in-vector (dataset-table-columns table))])
          (define n (vector-length column))
          (define mean (/ (for/sum ([v (in-vector column)]) v) n))
          (define sd (sqrt (/ (for/sum ([v (in-vector column)]) (sqr (- v mean))) n)))
          (for/list ([v (in-vector column)])
            (exact->inexact (/ (- v mean) sd))))
        (apply map list _))))

(define (make-cuml-eval)
  (parameterize ([sandbox-output 'string]
                 [sandbox-error-output 'string]
                 [sandbox-memory-limit #f]
                 [sandbox-eval-limits #f]
                 [sandbox-security-guard current-security-guard]
                 [sandbox-path-permissions '((exists "/"))])
    (apply make-base-eval
           #:lang 'racket/base
           '(require cuml
                     datasets
                     raft
                     raft/compat
                     racket/list
                     racket/math
                     threading)
           page-helpers)))
