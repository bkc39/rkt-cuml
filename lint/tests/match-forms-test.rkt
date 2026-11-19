#lang racket/base

(require (only-in racket/bool xor)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match match-define)
         (only-in rackunit check-equal? check-not-false check-true test-case)
         (only-in "../positional.rkt" positional-accesses)
         (only-in "../private/modules.rkt" with-temporary-directory write-module)
         (only-in "../private/patterns.rkt" match-forms match-layout)
         (only-in "private/oracle.rkt" alpha-copy? expansion-binders))

(define not-pattern-forms '(== define-match-expander failure-cont struct*))

(define match-syntax-exports
  (let ()
    (dynamic-require 'racket/match #f)
    (define-values (_variables syntaxes) (module->exports 'racket/match))
    (for*/list ([phase (in-list syntaxes)]
                #:when (eqv? (car phase) 0)
                [export (in-list (cdr phase))])
      (car export))))

(define (sample head layout pattern body)
  (match-define (match-layout before patterns scope) layout)
  (define lhs
    (if (eq? patterns 'several)
        (list pattern)
        pattern))
  (define leading
    (for/list ([one (in-list before)])
      (match one
        ['expression 'v]
        ['expressions '(v)]
        ['head '(f v)])))
  (match scope
    ['clause
     (define form `(,head ,@leading [,lhs ,body]))
     (if (memq 'head before)
         `(let ()
            ,form
            f)
         form)]
    ['definition
     `(let ()
        (,head ,lhs v)
        ,body)]
    [_ `(,head ([,lhs v]) ,body)]))

(define forms (sort (hash-keys match-forms) symbol<?))

(for ([name (in-list (remove-duplicates match-syntax-exports))])
  (test-case (format "racket/match's ~a is in the lint's table of pattern forms or named as taking none"
                     name)
    (check-not-false (xor (hash-has-key? match-forms name) (memq name not-pattern-forms)))))

(test-case "every form the table or the exclusions name is one racket/match exports"
  (for ([name (in-list (append forms not-pattern-forms))])
    (check-not-false (memq name match-syntax-exports) (symbol->string name))))

(for ([head (in-list forms)])
  (define layout (hash-ref match-forms head))
  (test-case (format "~a binds its patterns where the table says" head)
    (check-equal? (expansion-binders `(lambda (v) ,(sample head layout '(list a) '(binders a))))
                  '(a)))
  (test-case (format "the copy check renames a ~a pattern's binders" head)
    (check-true (alpha-copy? `(define (g v) ,(sample head layout '(list a) '(list a)))
                             `(define (g v) ,(sample head layout '(list b) '(list b)))))))

(test-case "no-positional-access reads every form's patterns from the table"
  (with-temporary-directory
   (root "match-forms-~a")
   (define files
     (for/list ([head (in-list forms)]
                [index (in-naturals)])
       (define layout (hash-ref match-forms head))
       (define refused (build-path root (format "form-~a-refused.rkt" index)))
       (define binds (build-path root (format "form-~a-binds.rkt" index)))
       (write-module refused
                     "#lang racket/base"
                     `((require racket/list racket/match)
                       (define (g v) ,(sample head layout '(app second a) 'a))))
       (write-module binds
                     "#lang racket/base"
                     `((require racket/list racket/match)
                       (define (g v) ,(sample head layout '(list second) 'second))))
       (list head refused binds)))
   (define found (map car (positional-accesses (list root) '(second))))
   (for ([one (in-list files)])
     (match-define (list head refused binds) one)
     (check-not-false (member refused found) (format "~a: (app second a) is refused" head))
     (check-equal? (member binds found) #f (format "~a: a pattern variable named second binds" head)))))
