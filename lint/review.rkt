#lang racket/base

(require (only-in racket/list append-map)
         (only-in review/ext pop-scope push-scope recur track-binding)
         ;; whole-module: its syntax classes come with it
         syntax/parse/define
         ;; whole-module: its syntax classes come with it
         syntax/parse/pre)

(provide review-syntax
         should-review?)

(define-syntax-parse-rule (with-scope body:expr ...+)
  (dynamic-wind push-scope
                (lambda ()
                  body ...)
                pop-scope))

(define (review-scoped-body leading body) ;; noqa
  (for-each recur leading)
  (with-scope (for-each recur body)))

(define-syntax-class reviewed-form
  #:datum-literals (define-syntax-parse-rule quasiquote
                     quote
                     test-gpu
                     test-twin
                     test-unless-skipped
                     test-without-gpu)
  (pattern (quote (_ . _))
    #:attr review void)
  (pattern (quasiquote datum)
    #:attr review (lambda () (for-each recur (unquoted #'datum 1))))
  (pattern (define-syntax-parse-rule (name:id . _)
             _ ...)
    #:attr review (lambda () (track-binding #'name)))
  (pattern (test-unless-skipped reason name body ...)
    #:attr review (lambda () (review-scoped-body (list #'reason #'name) (attribute body))))
  (pattern ((~or* test-gpu test-twin test-without-gpu) name body ...)
    #:attr review (lambda () (review-scoped-body (list #'name) (attribute body)))))

(define (should-review? stx)
  (syntax-parse stx
    [_:reviewed-form #t]
    [_ #f]))

(define (review-syntax stx)
  (syntax-parse stx
    [form:reviewed-form ((attribute form.review))]))

(define (unquoted stx depth)
  (syntax-parse stx
    #:datum-literals (quasiquote unquote unquote-splicing)
    [((~or* unquote unquote-splicing) e)
     #:when (= depth 1)
     (list #'e)]
    [((~or* unquote unquote-splicing) e) (unquoted #'e (sub1 depth))]
    [(quasiquote e) (unquoted #'e (add1 depth))]
    [(a . d) (append (unquoted #'a depth) (unquoted #'d depth))]
    [#(e ...) (append-map (lambda (e) (unquoted e depth)) (syntax->list #'(e ...)))]
    [_ (append-map (lambda (e) (unquoted e depth)) (quasiquoted-parts (syntax-e stx)))]))

(define (quasiquoted-parts datum)
  (cond
    [(box? datum) (list (unbox datum))]
    [(hash? datum) (hash-map datum (lambda (_key value) value) #t)]
    [(prefab-struct-key datum) (cdr (vector->list (struct->vector datum)))]
    [else '()]))
