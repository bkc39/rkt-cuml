#lang racket/base

(require (only-in racket/file find-files)
         (only-in racket/list append-map)
         (only-in racket/match match)
         (only-in "private/modules.rkt" read-module rkt?))

(provide positional-accesses)

(define (literal-index? one)
  (exact-nonnegative-integer? (syntax-e one)))

(define (accesses one refused)
  (cond
    [(and (identifier? one) (memq (syntax-e one) refused))
     (list (cons (syntax-e one) (syntax-line one)))]
    [(and (syntax? one)
          (match (syntax->list one)
            [(list (? identifier? head) _ (? literal-index?)) (eq? (syntax-e head) 'list-ref)]
            [_ #f]))
     (cons (cons 'list-ref (syntax-line one)) (accesses (syntax-e one) refused))]
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
            (if (eq? name 'list-ref) "list-ref with a literal index" name)
            (string-join (append (map symbol->string refused) '("list-ref with a literal index"))
                         ", ")))
  (unless (null? found)
    (exit 1)))
