#lang racket/base

(require (for-label cuml
                    datasets/core
                    (except-in pict pict)
                    plot
                    racket/base
                    racket/format
                    racket/function
                    racket/list
                    racket/match
                    racket/math
                    raft
                    threading))

(require (for-syntax racket/base
                     (only-in compiler/cm-accomplice register-external-module)
                     (only-in syntax/strip-context replace-context)
                     (only-in "guide/source.rkt" page-helpers-path page-helpers-source))
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
         (for-label (all-from-out cuml
                                  datasets/core
                                  pict
                                  plot
                                  racket/base
                                  racket/format
                                  racket/function
                                  racket/list
                                  racket/match
                                  racket/math
                                  raft
                                  threading))
         make-cuml-eval
         page-helpers-block
         page-require-block)

(define-syntax-parse-rule (define-page-helpers require-form:id
                                               require-block:id
                                               helper-forms:id
                                               helpers-block:id)
  #:do [(register-external-module page-helpers-path)
        (define-values (shown-require shown-helpers) (page-helpers-source))]
  #:with shown (replace-context #'require-form shown-require)
  #:with (form ...) (map (lambda (helper) (replace-context #'require-form helper)) shown-helpers)
  (begin
    (define require-form 'shown)
    (define require-block (racketblock shown))
    (define helper-forms '(form ...))
    (define helpers-block (racketblock form ...))))

(define-page-helpers page-require page-require-block page-helpers page-helpers-block)

(define (make-cuml-eval)
  (parameterize ([sandbox-output 'string]
                 [sandbox-error-output 'string]
                 [sandbox-memory-limit #f]
                 [sandbox-eval-limits #f]
                 [sandbox-security-guard current-security-guard]
                 [sandbox-path-permissions '((exists "/"))])
    (apply make-base-eval #:lang 'racket/base page-require page-helpers)))
