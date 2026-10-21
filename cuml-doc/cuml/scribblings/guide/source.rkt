#lang racket/base

(require (only-in racket/list partition)
         (only-in racket/match match)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in syntax/modread with-module-reading-parameterization))

(provide page-helpers-path
         page-helpers-source)

(define-runtime-path page-helpers-path "helpers.rkt")

(define (head form)
  (match (syntax->list form)
    [(cons name _) (syntax-e name)]
    [_ #f]))

(define (page-helpers-source)
  (define module-form
    (call-with-input-file page-helpers-path
                          (lambda (in)
                            (port-count-lines! in)
                            (with-module-reading-parameterization
                             (lambda () (read-syntax page-helpers-path in))))))
  (match (syntax->list module-form)
    [(list _ _ _ body)
     (define-values (requires definitions)
       (partition (lambda (form) (eq? (head form) 'require))
                  (filter (lambda (form) (not (eq? (head form) 'provide)))
                          (cdr (syntax->list body)))))
     (match requires
       [(list require-form) (values require-form definitions)])]))
