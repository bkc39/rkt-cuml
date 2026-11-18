#lang racket/base

(require (only-in racket/file find-files)
         (only-in racket/list append-map)
         (only-in "private/modules.rkt" read-module rkt?))

(provide positional-accesses)

(define (accesses one refused)
  (cond
    [(and (identifier? one) (memq (syntax-e one) refused))
     (list (cons (syntax-e one) (syntax-line one)))]
    [(syntax? one) (accesses (syntax-e one) refused)]
    [(pair? one) (append (accesses (car one) refused) (accesses (cdr one) refused))]
    [(vector? one) (append-map (lambda (item) (accesses item refused)) (vector->list one))]
    [else '()]))

(define (positional-accesses roots refused)
  (for*/list ([root (in-list roots)]
              [file (in-list (sort (find-files rkt? root) string<? #:key path->string))]
              [found (in-list (accesses (read-module file) refused))])
    (list file (cdr found) (car found))))

(module+ main
  (require (only-in racket/match match-define)
           (only-in racket/string string-join))
  (match-define (cons refused-text roots) (vector->list (current-command-line-arguments)))
  (define refused (map string->symbol (regexp-split #rx"," refused-text)))
  (define found (positional-accesses roots refused))
  (for ([one (in-list found)])
    (match-define (list file line name) one)
    (printf "~a:~a: ~a: read a test table's rows with match-define or match-lambda, not ~a\n"
            file
            line
            name
            (string-join (map symbol->string refused) ", ")))
  (unless (null? found)
    (exit 1)))
