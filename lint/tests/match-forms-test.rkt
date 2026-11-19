#lang racket/base

(require (only-in racket/bool xor)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match match* match-define)
         (only-in rackunit check-equal? check-false check-not-false check-true test-case)
         (only-in "../positional.rkt" positional-accesses)
         (only-in "../private/modules.rkt" with-temporary-directory write-module)
         (only-in "../private/patterns.rkt"
                  match-forms
                  match-layout
                  match-layout-before
                  match-layout-patterns
                  match-layout-scope)
         (only-in "private/oracle.rkt" alpha-copy? expansion-binders racket-value rename))

(define not-pattern-forms '(== define-match-expander failure-cont struct*))

(define match-syntax-exports
  (let ()
    (dynamic-require 'racket/match #f)
    (define-values (_variables syntaxes) (module->exports 'racket/match))
    (for*/list ([phase (in-list syntaxes)]
                #:when (eqv? (car phase) 0)
                [export (in-list (cdr phase))])
      (car export))))

(define (several? layout)
  (eq? (match-layout-patterns layout) 'several))

(define (clause? layout)
  (eq? (match-layout-scope layout) 'clause))

(define (clause-lhs layout pattern [other 'c])
  (if (several? layout)
      (list other pattern)
      pattern))

(define (sample-value layout)
  (if (several? layout)
      '(values v v)
      'v))

(define (clause-form head layout clauses)
  (define before (match-layout-before layout))
  (define leading
    (for/list ([one (in-list before)])
      (match* (one (several? layout))
        [('expression _) (sample-value layout)]
        [('expressions #t) '(v v)]
        [('expressions #f) '(v)]
        [('head #t) '(f v w)]
        [('head #f) '(f v)])))
  (define form `(,head ,@leading ,@clauses))
  (if (memq 'head before)
      `(let ()
         ,form
         f)
      form))

(define (sample head layout pattern body #:other [other 'c] #:failure [failure #f])
  (define lhs (clause-lhs layout pattern other))
  (match (match-layout-scope layout)
    ['clause
     (clause-form head
                  layout
                  (list (if failure
                            `[,lhs (=> ,failure) ,body]
                            `[,lhs ,body])))]
    ['definition
     `(let ()
        (,head ,lhs ,(sample-value layout))
        ,body)]
    [_ `(,head ([,lhs ,(sample-value layout)]) ,body)]))

(define (sample-binders layout)
  (append (if (several? layout)
              '(a c)
              '(a))
          (if (clause? layout)
              '(k)
              '())))

(define forms (sort (hash-keys match-forms) symbol<?))

(for ([name (in-list (remove-duplicates match-syntax-exports))])
  (test-case (format
              "racket/match's ~a is in the lint's table of pattern forms or named as taking none"
              name)
    (check-not-false (xor (hash-has-key? match-forms name) (memq name not-pattern-forms)))))

(test-case "every form the table or the exclusions name is one racket/match exports"
  (for ([name (in-list (append forms not-pattern-forms))])
    (check-not-false (memq name match-syntax-exports) (symbol->string name))))

(for ([head (in-list forms)])
  (define layout (hash-ref match-forms head))
  (test-case (format "~a binds its patterns where the table says" head)
    (check-equal? (expansion-binders
                   `(lambda (v w) ,(sample head layout '(list a) '(binders a c k list) #:failure 'k)))
                  (sample-binders layout)))
  (test-case (format "the copy check renames a ~a pattern's binders" head)
    (check-true (alpha-copy?
                 `(define (g v)
                    ,(sample head layout '(list a) `(list ,@(sample-binders layout)) #:failure 'k))
                 `(define (g v)
                    ,(sample head
                             layout
                             '(list b)
                             `(list ,@(rename (sample-binders layout)
                                              '((a . b) (c . d) (k . j))))
                             #:other 'd
                             #:failure 'j))))))

(define (scope-sample head layout #:backward? backward? #:formal formal #:binder binder #:read read)
  (define (lhs name)
    (if (several? layout)
        (list name)
        name))
  (define bindings
    (if backward?
        `([,(lhs 'b) (lambda () ,read)] [,(lhs binder) 1])
        `([,(lhs binder) 1] [,(lhs 'b) ,read])))
  `(lambda (,formal)
     (,head ,bindings
            (list ,binder
                  ,(if backward?
                       '(b)
                       'b)))))

(define (reads-binder? head layout backward?)
  (match-define (list _ seen)
    (racket-value
     (list (scope-sample head layout #:backward? backward? #:formal 'a #:binder 'a #:read 'a) 2)))
  (eqv? seen 1))

(define (racket-scope head layout)
  (match* ((reads-binder? head layout #f) (reads-binder? head layout #t))
    [(#f #f) 'parallel]
    [(#t #f) 'sequential]
    [(#t #t) 'recursive]
    [(_ _) 'unknown]))

(for ([head (in-list forms)]
      #:when (memq (match-layout-scope (hash-ref match-forms head)) '(parallel sequential recursive)))
  (define layout (hash-ref match-forms head))
  (test-case (format "~a scopes its bindings as Racket evaluates them" head)
    (check-equal? (match-layout-scope layout) (racket-scope head layout)))
  (for ([backward? (in-list '(#f #t))])
    (define (procedure formal binder read)
      `(define g
         ,(scope-sample head
                        layout
                        #:backward? backward?
                        #:formal formal
                        #:binder binder
                        #:read read)))
    (define original (procedure 'a 'a 'a))
    (test-case (format "the copy check reads a ~a right-hand side ~a its binder as Racket does"
                       head
                       (if backward? "before" "after"))
      (define binder? (reads-binder? head layout backward?))
      (check-true (alpha-copy? original (procedure 'z 'a (if binder? 'a 'z))) "the formal renamed")
      (check-true (alpha-copy? original (procedure 'a 'y (if binder? 'y 'a))) "the binder renamed")
      (check-false (alpha-copy? original (procedure 'q 'z (if binder? 'q 'z)))
                   "the right-hand side read the other way"))))

(define (clause-answer head layout clauses)
  (define form (clause-form head layout clauses))
  (racket-value `((lambda (k v)
                    ,(match (match-layout-before layout)
                       [(or '() '(head))
                        `(,form v
                                ,@(if (several? layout)
                                      '(v)
                                      '()))]
                       [_ form]))
                  'formal
                  1)))

(for ([head (in-list forms)]
      #:when (clause? (hash-ref match-forms head)))
  (define layout (hash-ref match-forms head))
  (define (lhs pattern)
    (clause-lhs layout pattern))
  (define (procedure formal clauses)
    `(define (g ,@(if formal
                      (list formal)
                      '())
                v)
       ,(clause-form head layout clauses)))
  (define (copy-as-racket-says? copy? one other message)
    (check-equal? (alpha-copy? one other) copy? message))
  (test-case (format "the copy check scopes a ~a pattern outside its clause's failure id" head)
    (define formal?
      (eq? (clause-answer head
                          layout
                          `([,(lhs '(? (lambda (x) (eq? k 'formal)))) (=> k) 'formal] [,(lhs '_)
                                                                                       'failure]))
           'formal))
    (define (probe formal read)
      (procedure formal `([,(lhs `(? ,read)) (=> k) 1] [,(lhs '_) 2])))
    (copy-as-racket-says? formal? (probe 'k 'k) (probe 'z 'z) "the formal renamed with its read")
    (copy-as-racket-says? (not formal?) (probe 'k 'k) (probe 'z 'k) "the formal renamed alone"))
  (test-case (format "the copy check scopes a ~a failure id inside its own clause" head)
    (define formal? (eq? (clause-answer head layout `([,(lhs '_) (=> k) (k)] [,(lhs '_) k])) 'formal))
    (define (probe formal read)
      (procedure formal `([,(lhs '_) (=> k) (k)] [,(lhs '_) ,read])))
    (copy-as-racket-says? formal? (probe 'k 'k) (probe 'z 'z) "the formal renamed with its read")
    (copy-as-racket-says? (not formal?) (probe 'k 'k) (probe 'z 'k) "the formal renamed alone"))
  (test-case (format "the copy check reads a ~a #:when after the failure id as Racket does" head)
    (define failure?
      (eq? (clause-answer head
                          layout
                          `([,(lhs '_) (=> k) #:when (procedure? k) 'failure] [,(lhs '_) 'formal]))
           'failure))
    (define (probe failure read)
      (procedure #f `([,(lhs '(list a)) (=> ,failure) #:when (procedure? ,read) a] [,(lhs '_) 1])))
    (copy-as-racket-says? failure?
                          (probe 'k 'k)
                          (probe 'j 'j)
                          "the failure id renamed with its read"))
  (define (shadowing binder failure)
    (procedure #f `([,(lhs `(list ,binder)) (=> ,failure) (add1 k)])))
  (for ([renamed (in-list '(failure pattern))])
    (test-case (format "the copy check lets a ~a pattern variable shadow a failure id: ~a renamed"
                       head
                       renamed)
      (define pattern?
        (eq? (clause-answer head layout `([,(lhs 'k) (=> k) (if (procedure? k) 'failure 'pattern)]))
             'pattern))
      (match renamed
        ['failure
         (copy-as-racket-says? pattern? (shadowing 'k 'k) (shadowing 'k 'j) "the failure id renamed")]
        ['pattern
         (copy-as-racket-says? (not pattern?)
                               (shadowing 'k 'k)
                               (shadowing 'j 'k)
                               "the pattern variable renamed")]))))

(test-case "no-positional-access reads every form's patterns from the table"
  (with-temporary-directory
   (root "match-forms-~a")
   (define files
     (for/list ([head (in-list forms)]
                [index (in-naturals)])
       (define layout (hash-ref match-forms head))
       (define refused (build-path root (format "form-~a-refused.rkt" index)))
       (define binds (build-path root (format "form-~a-binds.rkt" index)))
       (define fails (build-path root (format "form-~a-fails.rkt" index)))
       (write-module refused
                     "#lang racket/base"
                     `((require racket/list
                                racket/match)
                       (define (g v)
                         ,(sample head layout '(app second a) 'a))))
       (write-module binds
                     "#lang racket/base"
                     `((require racket/list
                                racket/match)
                       (define (g v)
                         ,(sample head layout '(list second) 'second))))
       (write-module fails
                     "#lang racket/base"
                     `((require racket/list
                                racket/match)
                       (define (g v)
                         ,(sample head layout '(list a) '(list a (second)) #:failure 'second))))
       (list head refused binds fails)))
   (define found (map car (positional-accesses (list root) '(second))))
   (for ([one (in-list files)])
     (match-define (list head refused binds fails) one)
     (check-not-false (member refused found) (format "~a: (app second a) is refused" head))
     (check-equal? (member binds found) #f (format "~a: a pattern variable named second binds" head))
     (check-equal? (and (member fails found) #t)
                   (not (clause? (hash-ref match-forms head)))
                   (format "~a: a failure id named second binds, and only a clause has one" head)))))
