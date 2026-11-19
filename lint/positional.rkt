#lang racket/base

(require (only-in racket/file find-files)
         (only-in racket/list append-map)
         (only-in racket/match match match*)
         (only-in "private/modules.rkt" read-module rkt?)
         (only-in "private/patterns.rkt"
                  binding-clauses
                  clause-ids
                  fold-head?
                  for-head?
                  match-form
                  match-form-parts
                  match-layout
                  match-parts
                  pattern-names
                  unquoted-parts))

(provide positional-accesses)

(define (literal-index? one)
  (match (syntax->datum one)
    [(or (? exact-nonnegative-integer?) (list (or 'quote 'quasiquote) (? exact-nonnegative-integer?)))
     #t]
    [_ #f]))

(define (formal-names formals)
  (match formals
    [(? symbol? rest) (list rest)]
    [(cons (? keyword?) more) (formal-names more)]
    [(cons (? symbol? one) more) (cons one (formal-names more))]
    [(cons (list (? symbol? one) _) more) (cons one (formal-names more))]
    [(cons _ more) (formal-names more)]
    [_ '()]))

(define (head-names head)
  (match head
    [(cons (? symbol? name) formals) (cons name (formal-names formals))]
    [(cons (? pair? inner) formals) (append (head-names inner) (formal-names formals))]
    [_ '()]))

(define (clause-names clauses)
  (append-map clause-ids (binding-clauses clauses)))

(define (match-form-names layout items)
  (match (match-form-parts layout items)
    [(match-parts leading groups _)
     (append (match* (layout leading)
               [((match-layout (list 'head) _ _) (list head)) (head-names head)]
               [(_ _) '()])
             (append-map (lambda (group) (append-map pattern-names (car group))) groups))]
    [#f '()]))

(define (bound-names datum)
  (match datum
    [(cons (or 'quote 'quasiquote) _) '()]
    [(list* (or 'define 'define-values) target rest)
     (append (match target
               [(? symbol? name) (list name)]
               [(? pair?) (head-names target)]
               [_ '()])
             (bound-names rest))]
    [(list* (or 'lambda 'λ) formals rest) (append (formal-names formals) (bound-names rest))]
    [(list* 'case-lambda clauses)
     (append-map (lambda (clause)
                   (match clause
                     [(cons formals rest) (append (formal-names formals) (bound-names rest))]
                     [_ '()]))
                 clauses)]
    [(list* 'let (? symbol? loop) clauses rest)
     (append (list loop) (clause-names clauses) (bound-names clauses) (bound-names rest))]
    [(list* (or 'let 'let* 'letrec 'letrec* 'let-values 'let*-values 'letrec-values) clauses rest)
     (append (clause-names clauses) (bound-names clauses) (bound-names rest))]
    [(cons (app match-form (? values layout)) items)
     (append (match-form-names layout items) (bound-names items))]
    [(list* (? fold-head?) (? list? accumulators) (? list? clauses) rest)
     (append (clause-names accumulators)
             (clause-names clauses)
             (bound-names accumulators)
             (bound-names clauses)
             (bound-names rest))]
    [(list* (? for-head?) (? list? clauses) rest)
     (append (clause-names clauses) (bound-names clauses) (bound-names rest))]
    [(cons a d) (append (bound-names a) (bound-names d))]
    [_ '()]))

(define (head-of items)
  (match items
    [(cons (? identifier? head) _) (syntax-e head)]
    [_ #f]))

(define (accesses one refused)
  (define items (and (syntax? one) (syntax->list one)))
  (define head (head-of items))
  (cond
    [(and (identifier? one) (memq (syntax-e one) refused))
     (list (cons (syntax-e one) (syntax-line one)))]
    [(eq? head 'quote) '()]
    [(eq? head 'quasiquote)
     (append-map (lambda (inner) (accesses inner refused)) (unquoted-parts (cdr items)))]
    [(match-accesses head items refused)]
    [(struct-options head items)
     =>
     (lambda (options) (accesses options refused))]
    [(and (eq? head 'list-ref)
          (match items
            [(list _ _ (? literal-index?)) #t]
            [_ #f]))
     (cons (cons 'list-ref (syntax-line one)) (accesses (syntax-e one) refused))]
    [(syntax? one) (accesses (syntax-e one) refused)]
    [(pair? one) (append (accesses (car one) refused) (accesses (cdr one) refused))]
    [(vector? one) (append-map (lambda (item) (accesses item refused)) (vector->list one))]
    [else '()]))

(define (struct-options head items)
  (match* (head items)
    [('struct (list* _ (? identifier?) (? identifier?) (? syntax?) options)) options]
    [((or 'struct 'define-struct) (list* _ _ (? syntax?) options)) options]
    [(_ _) #f]))

(define (match-accesses head items refused)
  (match (match (match-form head)
           [#f #f]
           [layout (match-form-parts layout (cdr items) syntax->list)])
    [(match-parts leading groups tail)
     (append (accesses leading refused)
             (append-map (lambda (group)
                           (append (append-map (lambda (one) (pattern-accesses one refused)) (car group))
                                   (accesses (cdr group) refused)))
                         groups)
             (accesses tail refused))]
    [#f #f]))

(define (pattern-accesses one refused)
  (define items (syntax->list one))
  (define (each patterns)
    (append-map (lambda (pattern) (pattern-accesses pattern refused)) patterns))
  (match* ((head-of items) items)
    [('quote _) '()]
    [('quasiquote (cons _ template)) (each (unquoted-parts template #:pattern? #t))]
    [((or '? 'app 'regexp 'pregexp) (list* _ operator patterns))
     (append (accesses operator refused) (each patterns))]
    [('== (cons _ operands)) (accesses operands refused)]
    [('struct (list _ _ (app syntax->list (? list? patterns)))) (each patterns)]
    [('struct* (list _ _ (app syntax->list (? list? fields))))
     (each (for/list ([field (in-list fields)])
             (match (syntax->list field)
               [(list _ pattern) pattern]
               [_ field])))]
    [(_ (? list?)) (each items)]
    [(_ _) (accesses one refused)]))

(define (file-accesses file refused)
  (define module (read-module file))
  (define own (bound-names (syntax->datum module)))
  (accesses module (filter (lambda (name) (not (memq name own))) refused)))

(define (positional-accesses roots refused)
  (for*/list ([root (in-list roots)]
              [file (in-list (sort (find-files rkt? root) string<? #:key path->string))]
              [found (in-list (file-accesses file refused))])
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
