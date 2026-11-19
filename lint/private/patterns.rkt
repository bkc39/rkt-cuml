#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/list append-map split-at)
         (only-in racket/match define-match-expander match match* match-define))

(provide binding-clauses
         clause-ids
         ellipsis?
         fold-head?
         for-head?
         lhs-patterns
         match-form
         match-form-parts
         match-forms
         (struct-out match-layout)
         (struct-out match-parts)
         pattern-head
         pattern-names
         proper
         quasi-map
         unquoted-parts)

(struct match-layout (before patterns scope) #:transparent)

(struct match-parts (leading groups tail) #:transparent)

(define match-forms
  (let ([one-per-clause (lambda before (match-layout before 'one 'clause))]
        [several-per-clause (lambda before (match-layout before 'several 'clause))])
    (hasheq 'match
            (one-per-clause 'expression)
            'match/derived
            (one-per-clause 'expression 'expression)
            'match*
            (several-per-clause 'expressions)
            'match*/derived
            (several-per-clause 'expressions 'expression)
            'match/values
            (several-per-clause 'expression)
            'match-lambda
            (one-per-clause)
            'match-λ
            (one-per-clause)
            'match-lambda*
            (one-per-clause)
            'match-λ*
            (one-per-clause)
            'match-lambda**
            (several-per-clause)
            'match-λ**
            (several-per-clause)
            'define/match
            (several-per-clause 'head)
            'match-let
            (match-layout '() 'one 'parallel)
            'match-let*
            (match-layout '() 'one 'sequential)
            'match-letrec
            (match-layout '() 'one 'recursive)
            'match-let-values
            (match-layout '() 'several 'parallel)
            'match-let*-values
            (match-layout '() 'several 'sequential)
            'match-letrec-values
            (match-layout '() 'several 'recursive)
            'match-define
            (match-layout '() 'one 'definition)
            'match-define-values
            (match-layout '() 'several 'definition))))

(define (match-form head)
  (and (symbol? head) (hash-ref match-forms head #f)))

(define (datum-list one)
  (and (list? one) one))

(define (lhs-patterns layout lhs [as-list datum-list])
  (if (eq? (match-layout-patterns layout) 'several)
      (or (as-list lhs) '())
      (list lhs)))

(define (match-form-parts layout items [as-list datum-list])
  (match-define (match-layout before _ scope) layout)
  (define (group one)
    (match (as-list one)
      [(cons lhs rest) (cons (lhs-patterns layout lhs as-list) rest)]
      [_ (list '() one)]))
  (define-values (leading more)
    (if (and (list? items) (<= (length before) (length items)))
        (split-at items (length before))
        (values #f #f)))
  (match* (leading scope more)
    [(#f _ _) #f]
    [(_ 'clause _) (match-parts leading (map group more) '())]
    [(_ 'definition (cons lhs rest))
     (match-parts leading (list (cons (lhs-patterns layout lhs as-list) rest)) '())]
    [(_ 'definition _) #f]
    [(_ _ (cons (app as-list (? list? bindings)) body)) (match-parts leading (map group bindings) body)]
    [(_ _ _) #f]))

(define-match-expander pattern-head
  (lambda (_stx)
    #'(or 'list 'list* 'list-rest 'list-no-order 'mlist 'cons 'mcons 'vector 'box 'and 'or 'not 'var)))

(define (for-head? head)
  (and (symbol? head) (regexp-match? #rx"^for[*]?(/|$)" (symbol->string head))))

(define (fold-head? head)
  (and (for-head? head) (regexp-match? #rx"/(fold|foldr|lists)$" (symbol->string head))))

(define (proper datum)
  (match datum
    [(cons a d) (cons a (proper d))]
    ['() '()]
    [tail (list tail)]))

(define (datum-of one)
  (if (syntax? one)
      (syntax-e one)
      one))

(define (quasi-map template evaluate #:pattern? [pattern? #f])
  (let walk ([one template]
             [depth 1])
    (define (again item)
      (walk item depth))
    (define datum (datum-of one))
    (match datum
      [(list (app datum-of (and head (or 'unquote 'unquote-splicing))) inner)
       (list head
             (if (or pattern? (= depth 1))
                 (evaluate inner)
                 (walk inner (sub1 depth))))]
      [(list (app datum-of 'quasiquote) inner) (list 'quasiquote (walk inner (add1 depth)))]
      [(cons a d) (cons (again a) (again d))]
      [(? vector?) (list->vector (map again (vector->list datum)))]
      [(? box?) (box (again (unbox datum)))]
      [(? hash?) (hash-map/copy datum (lambda (key value) (values key (again value))))]
      [(app prefab-struct-key (? values key))
       (apply make-prefab-struct key (map again (cdr (vector->list (struct->vector datum)))))]
      [_ datum])))

(define (unquoted-parts template #:pattern? [pattern? #f])
  (define parts '())
  (quasi-map template
             (lambda (inner)
               (set! parts (cons inner parts))
               inner)
             #:pattern? pattern?)
  (reverse parts))

(define (ellipsis? datum)
  (and (symbol? datum)
       (regexp-match? #px"^(?:[.]{3}|_{3}|(?:[.]{2}|_{2})[0-9]+)$" (symbol->string datum))))

(define (hash-patterns entries)
  (match entries
    [(list* '#:rest pattern _) (list pattern)]
    [(cons (? keyword?) rest) (hash-patterns rest)]
    [(list* _ pattern rest) (cons pattern (hash-patterns rest))]
    [_ '()]))

(define (hash*-patterns entries)
  (match entries
    [(list* '#:rest pattern _) (list pattern)]
    [(cons (list* _ pattern _) rest) (cons pattern (hash*-patterns rest))]
    [(cons _ rest) (hash*-patterns rest)]
    [_ '()]))

(define (pattern-names pattern)
  (match pattern
    [(or '_ (? ellipsis?)) '()]
    [(? symbol? name) (list name)]
    [(cons (or 'quote '== 'not) _) '()]
    [(cons 'quasiquote template) (append-map pattern-names (unquoted-parts template #:pattern? #t))]
    [(list* (or '? 'app 'regexp 'pregexp) _ patterns) (append-map pattern-names (proper patterns))]
    [(list 'struct _ (? list? patterns)) (append-map pattern-names patterns)]
    [(list 'struct _ _) '()]
    [(list 'struct* _ (? list? fields))
     (append-map (lambda (field)
                   (match field
                     [(list _ pattern) (pattern-names pattern)]
                     [_ '()]))
                 fields)]
    [(cons 'hash entries) (append-map pattern-names (hash-patterns (proper entries)))]
    [(cons 'hash* entries) (append-map pattern-names (hash*-patterns (proper entries)))]
    [(cons 'hash-table entries)
     (append-map (lambda (entry) (append-map pattern-names (proper entry))) (proper entries))]
    [(cons (pattern-head) patterns) (append-map pattern-names (proper patterns))]
    [(cons (? symbol?) fields) (append-map pattern-names (proper fields))]
    [(? pair?) (append-map pattern-names (proper pattern))]
    [_ '()]))

(define (clause-ids clause)
  (match clause
    [(list (? symbol? name) _) (list name)]
    [(list (? list? names) _) (filter symbol? names)]
    [(? symbol? name) (list name)]
    [_ '()]))

(define (binding-clauses clauses)
  (match clauses
    [(list* (? keyword?) _ rest) (binding-clauses rest)]
    [(cons one rest) (cons one (binding-clauses rest))]
    [_ '()]))
