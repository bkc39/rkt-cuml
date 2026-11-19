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
                            (list lhs body))))]
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

(define (argument-list? head layout)
  (and (null? (match-layout-before layout))
       (not (several? layout))
       (equal? (racket-value `(,(clause-form head layout '([x x])) 'probe)) '(probe))))

(define (clause-answer head layout clauses value)
  (define form (clause-form head layout clauses))
  (racket-value `((lambda (k v)
                    ,(match* ((match-layout-before layout) (argument-list? head layout))
                       [(_ #t) `(apply ,form v)]
                       [((or '() '(head)) _)
                        `(,form v
                                ,@(if (several? layout)
                                      '(v)
                                      '()))]
                       [(_ _) form]))
                  'formal
                  ',value)))

(define shadowing-patterns
  '((k k (5)) ((list k) k (5))
              ((list k ...) k (5 6))
              ((list a k ...) a (5 6))
              ((or (list k) (vector k)) k (5))
              ((list-no-order 5 k) k (5 6))
              ((list k ..1) k (5))))

(define (shadowing-sample lhs failure name position)
  (match position
    ['body `([,lhs (=> ,failure) (if (procedure? ,name) 'failure 'pattern)] [_ 'unmatched])]
    ['when `([,lhs (=> ,failure) #:when (procedure? ,name) 'failure] [,lhs 'pattern] [_ 'unmatched])]))

(define when-reads
  '((([(list a) (=> next) #:when (positive? a) a] [_ 0]) ([(list b) (=> next) #:when (positive? b) b]
                                                          [_ 0])
                                                         5)
    (([(list k) (=> k) #:when (number? k) k] [_ 0]) ([(list j) (=> k) #:when (number? j) j] [_ 0]) 5)
    (([(list k) (=> k) #:when (number? k) 1] [_ 0]) ([(list j) (=> k) #:when (number? k) 1] [_ 0]) 0)
    (([(list k ...) (=> k) #:when (list? k) k] [_ 0]) ([(list j ...) (=> k) #:when (list? j) j] [_ 0])
                                                      (5))
    (([(list k ...) (=> k) #:when (list? k) 1] [_ 0]) ([(list j ...) (=> k) #:when (list? k) 1] [_ 0])
                                                      0)))

(for ([head (in-list forms)]
      #:when (clause? (hash-ref match-forms head)))
  (define layout (hash-ref match-forms head))
  (define (lhs pattern)
    (clause-lhs layout pattern))
  (define (each-lhs clauses)
    (for/list ([clause (in-list clauses)])
      (match clause
        [(cons pattern forms) (cons (lhs pattern) forms)])))
  (define (procedure formal clauses)
    `(define (g ,formal v)
       ,(clause-form head layout (each-lhs clauses))))
  (define (answer clauses value)
    (clause-answer head layout (each-lhs clauses) value))
  (define (check-scope sample value copy-when one other)
    (define found (answer sample value))
    (check-not-false (memq found '(formal failure pattern)) "the sample matches")
    (match-define (list one-message one-formal one-clauses) one)
    (match-define (list other-message other-formal other-clauses) other)
    (check-equal? (alpha-copy? (procedure 'k sample) (procedure one-formal one-clauses))
                  (eq? found copy-when)
                  one-message)
    (check-equal? (alpha-copy? (procedure 'k sample) (procedure other-formal other-clauses))
                  (not (eq? found copy-when))
                  other-message))
  (test-case (format "the copy check scopes a ~a pattern outside its clause's failure id" head)
    (define (sample read)
      `([(? (lambda (x) (eq? ,read 'formal))) (=> k) 'formal] [_ 'failure]))
    (check-scope (sample 'k)
                 '(1)
                 'formal
                 (list "the formal renamed with its read" 'z (sample 'z))
                 (list "the formal renamed alone" 'z (sample 'k))))
  (test-case (format "the copy check scopes a ~a failure id inside its own clause" head)
    (define (sample read)
      `([_ (=> k) (k)] [_ (if (eq? ,read 'formal) 'formal 'failure)]))
    (check-scope (sample 'k)
                 '(1)
                 'formal
                 (list "the formal renamed with its read" 'z (sample 'z))
                 (list "the formal renamed alone" 'z (sample 'k))))
  (test-case (format "the copy check reads a ~a #:when after the failure id as Racket does" head)
    (define (sample failure read)
      `([_ (=> ,failure) #:when (procedure? ,read) 'failure] [_ 'formal]))
    (check-scope (sample 'k 'k)
                 '(1)
                 'failure
                 (list "the failure id renamed with its read" 'k (sample 'j 'j))
                 (list "the failure id renamed alone" 'k (sample 'j 'k))))
  (for* ([row (in-list shadowing-patterns)]
         [position (in-list '(body when))])
    (match-define (list pattern name value) row)
    (test-case (format "the copy check lets a ~a pattern ~s's ~a win over a failure id in the ~a as Racket does"
                       head
                       pattern
                       name
                       position)
      (check-scope (shadowing-sample pattern name name position)
                   value
                   'pattern
                   (list "the failure id renamed" 'k (shadowing-sample pattern 'j name position))
                   (list "the pattern variable renamed"
                         'k
                         (shadowing-sample (rename pattern (list (cons name 'j))) name name position)))))
  (for ([row (in-list when-reads)]
        [index (in-naturals)])
    (match-define (list one other copied) row)
    (test-case (format "the copy check reads ~a #:when read ~a's pattern variables as Racket does" head index)
      (check-equal? (alpha-copy? (procedure 'k one) (procedure 'k other))
                    (equal? (answer one '(5)) copied)))))

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
