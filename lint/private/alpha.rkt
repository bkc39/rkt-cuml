#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/list append-map)
         (only-in racket/match define-match-expander match)
         (only-in "patterns.rkt"
                  binding-clauses
                  clause-ids
                  fold-head?
                  for-head?
                  pattern-head
                  pattern-names
                  proper
                  quasi-map))

(provide alpha-normal
         procedure-name)

(define-match-expander procedure-head
  (lambda (_stx)
    #'(or 'lambda
          'λ
          'case-lambda
          'match-lambda
          'match-lambda*
          'const
          'curry
          'curryr
          'compose
          'compose1
          'negate
          'conjoin
          'disjoin
          'thunk
          'thunk*)))

(define (definition-root head)
  (match head
    [(cons (? symbol? name) _) name]
    [(cons (? pair? inner) _) (definition-root inner)]
    [_ #f]))

(define (procedure-name datum)
  (match datum
    [(list* (or 'define 'define/who 'define/contract) (? pair? head) _ _) (definition-root head)]
    [(list 'define (? symbol? name) (cons (procedure-head) _)) name]
    [(list 'define/contract (? symbol? name) _ (cons (procedure-head) _)) name]
    [_ #f]))

(define (body-names forms)
  (append-map
   (lambda (form)
     (match form
       [(list* (or 'define 'define/who 'define/contract) (? pair? head) _)
        (list (definition-root head))]
       [(list* (or 'define 'define/contract) (? symbol? name) _) (list name)]
       [(list* 'define-values (? list? names) _) (filter symbol? names)]
       [(list* 'match-define pattern _) (pattern-names pattern)]
       [(list* 'match-define-values (? list? patterns) _) (append-map pattern-names patterns)]
       [_ '()]))
   forms))

(define (alpha-normal definition name free-key)
  (define count 0)
  (define (fresh!)
    (begin0 (string->symbol (format "_~a" count))
      (set! count (add1 count))))
  (define (bind env names)
    (for/fold ([env env]) ([one (in-list names)])
      (hash-set env one (fresh!))))
  (define (ref env one)
    (hash-ref env one (lambda () (free-key one))))
  (define (each env items)
    (if (and (list? items) (pair? (body-names items)))
        (body env items)
        (each-form env items)))
  (define (each-form env items)
    (match items
      [(cons a d) (cons (expr env a) (each-form env d))]
      [_ (expr env items)]))
  (define (formals env items)
    (match items
      ['() (values '() env)]
      [(? symbol? rest)
       (define inner (bind env (list rest)))
       (values (ref inner rest) inner)]
      [(list* '#:hyperparameters (? symbol? table) rest)
       (define-values (more inner) (formals env rest))
       (values (list* '#:hyperparameters (ref env table) more) inner)]
      [(cons (? keyword? key) rest)
       (define-values (more inner) (formals env rest))
       (values (cons key more) inner)]
      [(cons (list (? symbol? one) default) rest)
       (define bound (bind env (list one)))
       (define-values (more inner) (formals bound rest))
       (values (cons (list (ref bound one) (expr env default)) more) inner)]
      [(cons (? symbol? one) rest)
       (define bound (bind env (list one)))
       (define-values (more inner) (formals bound rest))
       (values (cons (ref bound one) more) inner)]
      [(cons other rest)
       (define-values (more inner) (formals env rest))
       (values (cons other more) inner)]))
  (define (definition-head env head)
    (match head
      [(cons (? symbol? one) items)
       (define-values (normal inner) (formals env items))
       (values (cons (ref env one) normal) inner)]
      [(cons (? pair? curried) items)
       (define-values (outer middle) (definition-head env curried))
       (define-values (normal inner) (formals middle items))
       (values (cons outer normal) inner)]))
  (define (body env forms)
    (define inner (bind env (body-names forms)))
    (map (lambda (form) (body-form inner form)) forms))
  (define (body-form env form)
    (match form
      [(list* (and definer (or 'define 'define/who)) (? pair? head) forms)
       #:when (not (hash-ref env definer #f))
       (define-values (normal inner) (definition-head env head))
       (list* (ref env definer) normal (body inner forms))]
      [(list* 'define/contract (? pair? head) contract forms)
       (define-values (normal inner) (definition-head env head))
       (list* (ref env 'define/contract) normal (expr env contract) (body inner forms))]
      [(list* (and definer (or 'define 'define/contract 'define-values)) target rest)
       #:when (not (hash-ref env definer #f))
       (list* (ref env definer) (each env target) (each env rest))]
      [(list* (and definer (or 'match-define 'match-define-values)) patterns rest)
       #:when (not (hash-ref env definer #f))
       (list* (ref env definer) (normal-pattern env patterns) (each env rest))]
      [_ (expr env form)]))
  (define (normal-pattern env datum)
    (match datum
      [(or '_ '... '..1 '___) datum]
      [(? symbol?) (ref env datum)]
      [(cons (or 'quote 'regexp 'pregexp) _) datum]
      [(cons 'quasiquote template)
       (cons 'quasiquote (quasi-map template (lambda (inner) (normal-pattern env inner))))]
      [(list* (and head (or '? 'app '==)) predicate patterns)
       (list* head
              (expr env predicate)
              (map (lambda (one) (normal-pattern env one)) (proper patterns)))]
      [(cons (and head (pattern-head)) patterns)
       (cons head (map (lambda (one) (normal-pattern env one)) (proper patterns)))]
      [(cons (? symbol? head) patterns)
       (cons (ref env head) (map (lambda (one) (normal-pattern env one)) (proper patterns)))]
      [(? pair?) (map (lambda (one) (normal-pattern env one)) (proper datum))]
      [_ datum]))
  (define (clauses env items #:several? [several? #f])
    (for/list ([clause (in-list (proper items))])
      (match clause
        [(cons patterns forms)
         (define each-pattern
           (if (and several? (list? patterns))
               patterns
               (list patterns)))
         (define inner (bind env (append-map pattern-names each-pattern)))
         (define normal (map (lambda (one) (normal-pattern inner one)) each-pattern))
         (cons (if several?
                   normal
                   (car normal))
               (body inner forms))]
        [_ clause])))
  (define (guarded env items normal-test)
    (for/list ([clause (in-list (proper items))])
      (match clause
        [(cons test forms) (cons (normal-test test) (body env forms))]
        [_ clause])))
  (define (let-form env head rest)
    (match rest
      [(list* (? symbol? loop) (? list? bindings) forms)
       #:when (eq? head 'let)
       (define inits (map (lambda (one) (each env (cdr one))) (filter pair? bindings)))
       (define inner (bind (bind env (list loop)) (append-map clause-ids bindings)))
       (list* (ref inner loop)
              (map (lambda (one init) (cons (each inner (car one)) init)) bindings inits)
              (body inner forms))]
      [(list* (? list? bindings) forms)
       (define sequential?
         (memq head
               '(let* let*-values
                  letrec
                  letrec*
                  letrec-values
                  match-let*)))
       (define recursive?
         (memq head
               '(letrec letrec*
                  letrec-values)))
       (define matching?
         (memq head
               '(match-let match-let*
                  )))
       (define names
         (if matching?
             (lambda (one) (pattern-names (car one)))
             clause-ids))
       (define all
         (if recursive?
             (bind env (append-map names bindings))
             env))
       (define-values (normal inner)
         (for/fold ([normal '()]
                    [scope all]
                    #:result (values (reverse normal) scope))
                   ([one (in-list bindings)])
           (define init
             (each (if sequential? scope all)
                   (if (pair? one)
                       (cdr one)
                       '())))
           (define bound
             (if recursive?
                 scope
                 (bind scope (names one))))
           (define target
             (if matching?
                 (normal-pattern bound (car one))
                 (each bound
                       (if (pair? one)
                           (car one)
                           one))))
           (values (cons (cons target init) normal) bound)))
       (list* normal (body inner forms))]
      [_ (each env rest)]))
  (define (for-form env head rest)
    (match rest
      [(list* accumulators (? list? items) forms)
       #:when (fold-head? head)
       (define accumulated (bind env (append-map clause-ids (binding-clauses accumulators))))
       (define-values (normal inner) (for-clauses accumulated items))
       (list* (each accumulated accumulators) normal (body inner forms))]
      [(list* (? list? items) forms)
       (define-values (normal inner) (for-clauses env items))
       (list* normal (body inner forms))]
      [_ (each env rest)]))
  (define (for-clauses env items)
    (for/fold ([normal '()]
               [scope env]
               #:result (values (reverse normal) scope))
              ([one (in-list items)])
      (match one
        [(list target sequence)
         (define bound (bind scope (clause-ids one)))
         (values (cons (list (each bound target) (expr scope sequence)) normal) bound)]
        [_ (values (cons (expr scope one) normal) scope)])))
  (define (expr env datum)
    (match datum
      [(? symbol?) (ref env datum)]
      [(cons (? symbol? head) rest)
       #:when (not (hash-ref env head #f))
       (define key (ref env head))
       (cons key
             (match head
               [(or 'quote 'quote-syntax 'syntax) rest]
               [(or 'quasiquote 'quasisyntax) (quasi-map rest (lambda (inner) (expr env inner)))]
               [(or 'lambda 'λ)
                (match rest
                  [(cons items forms)
                   (define-values (normal inner) (formals env items))
                   (cons normal (body inner forms))]
                  [_ (each env rest)])]
               ['case-lambda
                (for/list ([clause (in-list (proper rest))])
                  (match clause
                    [(cons items forms)
                     (define-values (normal inner) (formals env items))
                     (cons normal (body inner forms))]
                    [_ clause]))]
               [(or 'match 'match*)
                (match rest
                  [(cons subject items)
                   (cons (expr env subject) (clauses env items #:several? (eq? head 'match*)))]
                  [_ (each env rest)])]
               [(or 'match-lambda 'match-lambda*) (clauses env rest)]
               [(or 'let
                    'let*
                    'letrec
                    'letrec*
                    'let-values
                    'let*-values
                    'letrec-values
                    'match-let
                    'match-let*)
                (let-form env head rest)]
               [(? for-head?) (for-form env head rest)]
               [(or 'when 'unless)
                (match rest
                  [(cons test forms) (cons (expr env test) (body env forms))]
                  [_ (each env rest)])]
               ['cond (guarded env rest (lambda (test) (expr env test)))]
               ['case
                (match rest
                  [(cons subject items) (cons (expr env subject) (guarded env items values))]
                  [_ (each env rest)])]
               [(or 'parameterize 'parameterize* 'with-handlers 'with-handlers*)
                (match rest
                  [(cons (? list? bindings) forms)
                   (cons (map (lambda (one) (each-form env one)) bindings) (body env forms))]
                  [_ (each env rest)])]
               [_ (each env rest)]))]
      [(? pair?) (each env datum)]
      [_ datum]))
  (body-form (bind (hasheq) (list name)) definition))
