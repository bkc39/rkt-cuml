#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/list append-map split-at)
         (only-in racket/match define-match-expander match match*)
         (only-in "patterns.rkt"
                  binding-clauses
                  clause-ids
                  direct-pattern-names
                  ellipsis?
                  failure-forms
                  fold-head?
                  for-head?
                  lhs-patterns
                  match-form
                  match-layout
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
          (? procedure-form?)
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

(define (procedure-form? head)
  (match (match-form head)
    [(match-layout '() _ 'clause) #t]
    [_ #f]))

(define (lhs-names layout lhs)
  (append-map pattern-names (lhs-patterns layout lhs)))

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
  (append-map (lambda (form)
                (match form
                  [(list* (or 'define 'define/who 'define/contract) (? pair? head) _)
                   (list (definition-root head))]
                  [(list* (or 'define 'define/contract) (? symbol? name) _) (list name)]
                  [(list* 'define-values (? list? names) _) (filter symbol? names)]
                  [(list* (app match-form (and layout (match-layout _ _ 'definition))) lhs _)
                   (lhs-names layout lhs)]
                  [(list* (app match-form (match-layout '(head) _ _)) (? pair? head) _)
                   (list (definition-root head))]
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
      [(cons (and definer (app match-form (? values layout))) rest)
       #:when (not (hash-ref env definer #f))
       (cons (ref env definer) (match-form-normal env layout rest))]
      [_ (expr env form)]))
  (define (normal-pattern env datum)
    (define (normal one)
      (normal-pattern env one))
    (match datum
      [(or '_ (? ellipsis?)) datum]
      [(? symbol?) (ref env datum)]
      [(cons 'quote _) datum]
      [(cons 'quasiquote template) (cons 'quasiquote (quasi-map template normal #:pattern? #t))]
      [(cons '== exprs) (cons '== (map (lambda (one) (expr env one)) (proper exprs)))]
      [(list* (and head (or '? 'app 'regexp 'pregexp)) predicate patterns)
       (list* head (expr env predicate) (map normal (proper patterns)))]
      [(list 'struct* id (? list? fields))
       (list 'struct*
             (ref env id)
             (for/list ([field (in-list fields)])
               (match field
                 [(list name one) (list name (normal one))]
                 [_ field])))]
      [(list 'struct id (? list? patterns)) (list 'struct (ref env id) (map normal patterns))]
      [(list 'struct id other) (list 'struct (ref env id) other)]
      [(cons 'hash entries) (cons 'hash (hash-entries env (proper entries)))]
      [(cons 'hash* entries) (cons 'hash* (hash*-entries env (proper entries)))]
      [(cons 'hash-table entries)
       (cons 'hash-table
             (for/list ([entry (in-list (proper entries))])
               (if (pair? entry)
                   (map normal (proper entry))
                   entry)))]
      [(list 'var (? symbol? id)) (list 'var (ref env id))]
      [(cons 'not patterns)
       (define inner (bind env (append-map pattern-names (proper patterns))))
       (cons 'not (map (lambda (one) (normal-pattern inner one)) (proper patterns)))]
      [(cons (and head (pattern-head)) patterns) (cons head (map normal (proper patterns)))]
      [(cons (? symbol? head) patterns) (cons (ref env head) (map normal (proper patterns)))]
      [(? pair?) (map normal (proper datum))]
      [_ datum]))
  (define (hash-entries env entries)
    (match entries
      [(list* '#:rest one rest) (list* '#:rest (normal-pattern env one) rest)]
      [(cons (? keyword? key) rest) (cons key (hash-entries env rest))]
      [(list* key one rest) (list* (expr env key) (normal-pattern env one) (hash-entries env rest))]
      [_ entries]))
  (define (hash*-entries env entries)
    (match entries
      [(list* '#:rest one rest) (list* '#:rest (normal-pattern env one) rest)]
      [(cons (list* key one options) rest)
       (cons (list* (expr env key)
                    (normal-pattern env one)
                    (for/list ([option (in-list (proper options))])
                      (if (keyword? option)
                          option
                          (expr env option))))
             (hash*-entries env rest))]
      [(cons other rest) (cons other (hash*-entries env rest))]
      [_ entries]))
  (define (normal-lhs env layout lhs)
    (match* (layout lhs)
      [((match-layout _ 'several _) (? list?)) (map (lambda (one) (normal-pattern env one)) lhs)]
      [((match-layout _ 'several _) _) lhs]
      [(_ _) (normal-pattern env lhs)]))
  (define (match-form-normal env layout rest)
    (match* (layout rest)
      [((match-layout '(head) _ _) (cons (? pair? head) items))
       (define-values (normal inner) (definition-head env head))
       (cons normal (clauses inner items #:several? #t))]
      [((match-layout before patterns 'clause) _)
       #:when (and (list? rest) (<= (length before) (length rest)))
       (define-values (leading items) (split-at rest (length before)))
       (append (map (lambda (one) (expr env one)) leading)
               (clauses env items #:several? (eq? patterns 'several)))]
      [((match-layout _ _ 'definition) (cons lhs forms))
       (cons (normal-lhs env layout lhs) (each env forms))]
      [((match-layout _ _ (and scope (or 'parallel 'sequential 'recursive))) _)
       (let-form env
                 rest
                 #:sequential? (memq scope '(sequential recursive))
                 #:recursive? (eq? scope 'recursive)
                 #:names (lambda (one) (lhs-names layout (car one)))
                 #:target (lambda (bound one) (normal-lhs bound layout (car one))))]
      [(_ _) (each env rest)]))
  (define (clauses env items #:several? [several? #f])
    (for/list ([clause (in-list (proper items))])
      (match clause
        [(cons patterns forms)
         (define each-pattern
           (if (and several? (list? patterns))
               patterns
               (list patterns)))
         (define names (append-map pattern-names each-pattern))
         (define inner (bind env names))
         (define normal (map (lambda (one) (normal-pattern inner one)) each-pattern))
         (cons (if several?
                   normal
                   (car normal))
               (clause-body inner (append-map direct-pattern-names each-pattern) forms))]
        [_ clause])))
  (define (clause-body inner names forms)
    (match forms
      [(failure-forms id more)
       (define failing (bind inner (list id)))
       (define scope
         (for/fold ([scope failing]) ([name (in-list names)])
           (hash-set scope name (ref inner name))))
       (cons (list '=> (ref failing id)) (body scope more))]
      [_ (body inner forms)]))
  (define (guarded env items normal-test)
    (for/list ([clause (in-list (proper items))])
      (match clause
        [(cons test forms) (cons (normal-test test) (body env forms))]
        [_ clause])))
  (define (let-form env
                    rest
                    #:named? [named? #f]
                    #:sequential? sequential?
                    #:recursive? recursive?
                    #:names [names clause-ids]
                    #:target [target
                              (lambda (bound one)
                                (each bound
                                      (if (pair? one)
                                          (car one)
                                          one)))])
    (match rest
      [(list* (? symbol? loop) (? list? bindings) forms)
       #:when named?
       (define inits (map (lambda (one) (each env (cdr one))) (filter pair? bindings)))
       (define inner (bind (bind env (list loop)) (append-map clause-ids bindings)))
       (list* (ref inner loop)
              (map (lambda (one init) (cons (each inner (car one)) init)) bindings inits)
              (body inner forms))]
      [(list* (? list? bindings) forms)
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
           (values (cons (cons (target bound one) init) normal) bound)))
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
    (match items
      [(list* (? keyword? key) form rest)
       (define guard (expr env form))
       (define-values (normal inner) (for-clauses env rest))
       (values (list* key guard normal) inner)]
      [(cons (and one (list target sequence)) rest)
       (define bound (bind env (clause-ids one)))
       (define clause (list (each bound target) (expr env sequence)))
       (define-values (normal inner) (for-clauses bound rest))
       (values (cons clause normal) inner)]
      [(cons one rest)
       (define item (expr env one))
       (define-values (normal inner) (for-clauses env rest))
       (values (cons item normal) inner)]
      [_ (values '() env)]))
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
               [(app match-form (? values layout)) (match-form-normal env layout rest)]
               [(or 'let 'let* 'letrec 'letrec* 'let-values 'let*-values 'letrec-values)
                (let-form env
                          rest
                          #:named? (eq? head 'let)
                          #:sequential? (memq head
                                              '(let* let*-values
                                                 letrec
                                                 letrec*
                                                 letrec-values))
                          #:recursive? (memq head
                                             '(letrec letrec*
                                                letrec-values)))]
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
