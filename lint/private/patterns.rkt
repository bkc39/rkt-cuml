#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/list append-map)
         (only-in racket/match define-match-expander match))

(provide binding-clauses
         clause-ids
         fold-head?
         for-head?
         pattern-head
         pattern-names
         proper
         quasi-map
         unquoted-parts)

(define-match-expander pattern-head
  (lambda (_stx)
    #'(or 'list
          'list*
          'list-rest
          'list-no-order
          'cons
          'vector
          'box
          'and
          'or
          'not
          'hash-table
          'struct*)))

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

(define (quasi-map template evaluate)
  (let walk ([one template]
             [depth 1])
    (define (again item)
      (walk item depth))
    (define datum (datum-of one))
    (match datum
      [(list (app datum-of (and head (or 'unquote 'unquote-splicing))) inner)
       (list head
             (if (= depth 1)
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

(define (unquoted-parts template)
  (define parts '())
  (quasi-map template
             (lambda (inner)
               (set! parts (cons inner parts))
               inner))
  (reverse parts))

(define (pattern-names pattern)
  (match pattern
    [(or '_ '... '..1 '___) '()]
    [(? symbol? name) (list name)]
    [(cons (or 'quote 'regexp 'pregexp '==) _) '()]
    [(cons 'quasiquote template) (append-map pattern-names (unquoted-parts template))]
    [(list* (or '? 'app) _ patterns) (append-map pattern-names (proper patterns))]
    [(list 'struct* _ (? list? fields))
     (append-map (lambda (field)
                   (match field
                     [(list _ pattern) (pattern-names pattern)]
                     [_ '()]))
                 fields)]
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
