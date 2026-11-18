#lang racket/base

(require (only-in racket/file find-files)
         (only-in racket/list append-map remove-duplicates)
         (only-in racket/match match match-lambda)
         (only-in racket/string string-join)
         (only-in syntax/modread with-module-reading-parameterization))

(provide copied-definitions
         copied-definitions-in
         describe-copies)

(struct site (file line name) #:transparent)

(define lambda-heads '(lambda λ))

(define let-heads '(let let* letrec letrec* let-values let*-values letrec-values))

(define match-heads '(match match*))

(define match-clause-heads '(match-lambda match-lambda* match-let match-let*))

(define pattern-heads '(list list* cons vector struct* and or hash-table))

(define (for-head? head)
  (and (symbol? head) (regexp-match? #rx"^for[*]?(/|$)" (symbol->string head))))

(define (read-module file)
  (with-input-from-file file
                        (lambda ()
                          (port-count-lines! (current-input-port))
                          (with-module-reading-parameterization read-syntax))))

(define (module-forms file)
  (match (syntax-e (read-module file))
    [(list _ _ _ body)
     (match (syntax-e body)
       [(cons _ forms) forms]
       [_ '()])]
    [_ '()]))

(define (procedure-name datum)
  (match datum
    [(list* (or 'define 'define/who) (cons (? symbol? name) _) _ _) name]
    [(list 'define (? symbol? name) (cons (? (lambda (head) (memq head lambda-heads))) _)) name]
    [_ #f]))

(define (proper datum)
  (match datum
    [(cons a d) (cons a (proper d))]
    ['() '()]
    [tail (list tail)]))

(define (formal-names formals)
  (match formals
    [(? symbol? name) (list name)]
    [(list* '#:hyperparameters _ rest) (formal-names rest)]
    [(cons (? keyword?) rest) (formal-names rest)]
    [(cons (or (? symbol? name) (list (? symbol? name) _)) rest) (cons name (formal-names rest))]
    [(cons _ rest) (formal-names rest)]
    [_ '()]))

(define (clause-names clauses)
  (append-map (match-lambda
                [(cons (? symbol? name) _) (list name)]
                [(cons (? list? names) _) (filter symbol? names)]
                [_ '()])
              (if (list? clauses) clauses '())))

(define (pattern-names pattern)
  (match pattern
    [(or '_ '... '..1) '()]
    [(? symbol? name) (list name)]
    [(cons (or 'quote 'regexp 'pregexp) _) '()]
    [(list* (or '? 'app) _ patterns) (append-map pattern-names (proper patterns))]
    [(cons (? (lambda (head) (memq head pattern-heads))) patterns) (append-map pattern-names (proper patterns))]
    [(cons (? symbol?) fields) (append-map field-pattern-names (proper fields))]
    [(? pair?) (append-map pattern-names (proper pattern))]
    [_ '()]))

(define (field-pattern-names field)
  (match field
    [(list (? symbol?) pattern) (pattern-names pattern)]
    [_ (pattern-names field)]))

(define (clause-pattern-names clauses)
  (append-map (match-lambda
                [(cons pattern _) (pattern-names pattern)]
                [_ '()])
              (if (list? clauses) clauses '())))

(define (head-binders datum)
  (match datum
    [(list* (or 'define 'define/who) (cons name formals) _) (cons name (formal-names formals))]
    [(list* 'define (? symbol? name) _) (list name)]
    [(list* 'define-values (? list? names) _) (filter symbol? names)]
    [(list* (? (lambda (head) (memq head lambda-heads))) formals _) (formal-names formals)]
    [(cons 'case-lambda clauses) (append-map (lambda (clause) (formal-names (car clause))) (filter pair? clauses))]
    [(list* (? (lambda (head) (memq head let-heads))) (? symbol? name) clauses _) (cons name (clause-names clauses))]
    [(list* (? (lambda (head) (memq head let-heads))) clauses _) (clause-names clauses)]
    [(list* (or 'for/fold 'for*/fold 'for/foldr) accumulators clauses _)
     (append (clause-names accumulators) (clause-names clauses))]
    [(list* (? for-head?) clauses _) (clause-names clauses)]
    [(list* 'match-define pattern _) (pattern-names pattern)]
    [(list* (? (lambda (head) (memq head match-heads))) _ clauses) (clause-pattern-names clauses)]
    [(cons (or 'match-lambda 'match-lambda*) clauses) (clause-pattern-names clauses)]
    [(list* (or 'match-let 'match-let*) clauses _) (clause-pattern-names clauses)]
    [_ '()]))

(define (binders datum)
  (match datum
    [(? pair?) (append (head-binders datum) (append-map binders (proper datum)))]
    [(? vector?) (append-map binders (vector->list datum))]
    [_ '()]))

(define (struct-names name fields)
  (append (list name (symbol-append name '?) (symbol-append 'make- name))
          (for/list ([field (in-list fields)]
                     #:when (or (symbol? field) (and (pair? field) (symbol? (car field)))))
            (symbol-append name '- (if (pair? field) (car field) field)))))

(define (symbol-append . parts)
  (string->symbol (apply string-append (map symbol->string parts))))

(define (defined-names datum)
  (match datum
    [(list* (or 'struct 'define-struct) (? symbol? name) (? symbol?) (? list? fields) _) (struct-names name fields)]
    [(list* (or 'struct 'define-struct) (? symbol? name) (? list? fields) _) (struct-names name fields)]
    [(list* 'define-values (? list? names) _) (filter symbol? names)]
    [(list* (? definer?) (or (? symbol? name) (cons (? symbol? name) _)) _) (list name)]
    [_ '()]))

(define (definer? head)
  (and (symbol? head) (regexp-match? #rx"^define" (symbol->string head))))

(define (alpha-normal datum file-locals file)
  (define bound (remove-duplicates (binders datum) eq?))
  (define renamed (make-hasheq))
  (let rename ([one datum])
    (match one
      [(? symbol?)
       #:when (memq one bound)
       (hash-ref! renamed one (lambda () (string->symbol (format "_~a" (hash-count renamed)))))]
      [(? symbol?)
       #:when (memq one file-locals)
       (format "~a@~a" one file)]
      [(cons a d) (cons (rename a) (rename d))]
      [(? vector?) (list->vector (map rename (vector->list one)))]
      [_ one])))

(define (definitions-in file)
  (define forms (module-forms file))
  (define file-locals (append-map (lambda (form) (defined-names (syntax->datum form))) forms))
  (for*/list ([form (in-list forms)]
              [datum (in-value (syntax->datum form))]
              [name (in-value (procedure-name datum))]
              #:when name)
    (cons (alpha-normal datum file-locals file) (site file (syntax-line form) name))))

(define (site<? a b)
  (define fa (path->string (site-file a)))
  (define fb (path->string (site-file b)))
  (or (string<? fa fb) (and (string=? fa fb) (< (site-line a) (site-line b)))))

(define (copied-definitions files)
  (define by-body (make-hash))
  (for* ([file (in-list files)]
         [entry (in-list (definitions-in file))])
    (hash-update! by-body (car entry) (lambda (sites) (cons (cdr entry) sites)) '()))
  (sort (for*/list ([sites (in-hash-values by-body)]
                    [ordered (in-value (sort sites site<?))]
                    #:when (pair? (cdr (remove-duplicates (map site-file ordered)))))
          ordered)
        site<?
        #:key car))

(define (scanned? path)
  (define text (path->string path))
  (and (regexp-match? #rx"[.]rkt$" text) (not (regexp-match? #rx"/compiled/" text))))

(define (copied-definitions-in roots)
  (copied-definitions (sort (append-map (lambda (root) (find-files scanned? root)) roots)
                            string<?
                            #:key path->string)))

(define (describe-copies group)
  (string-join (for/list ([one (in-list group)])
                 (format "~a:~a: ~a" (site-file one) (site-line one) (site-name one)))
               "\n  "
               #:before-first "copied definition: the same body after renaming, in more than one file; keep one, in the family's shared module, and require it\n  "))

(module+ main
  (define pools (map (lambda (pool) (regexp-split #rx":" pool)) (vector->list (current-command-line-arguments))))
  (define groups (append-map copied-definitions-in pools))
  (for-each (lambda (group) (displayln (describe-copies group))) groups)
  (unless (null? groups)
    (exit 1)))
