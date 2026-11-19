#lang racket/base

(require (only-in racket/list append-map remove-duplicates)
         (only-in racket/set list->seteq)
         (only-in rackunit check-equal? test-case)
         (only-in "../private/patterns.rkt" pattern-names))

(define oracle-namespace (make-base-namespace))

(define (symbols datum)
  (cond
    [(symbol? datum) (list datum)]
    [(pair? datum) (append (symbols (car datum)) (symbols (cdr datum)))]
    [(vector? datum) (append-map symbols (vector->list datum))]
    [(box? datum) (symbols (unbox datum))]
    [(prefab-struct-key datum) (append-map symbols (cdr (vector->list (struct->vector datum))))]
    [else '()]))

(define (racket-binders pattern)
  (define candidates (remove-duplicates (symbols pattern)))
  (with-handlers ([(lambda (raised) (and (pair? raised) (eq? (car raised) 'binders))) cdr])
    (parameterize ([current-namespace oracle-namespace])
      (expand `(module oracle racket/base
                 (require racket/match
                          (for-syntax racket/base))
                 (struct point (a b))
                 (define-syntax (binders stx)
                   (raise (cons 'binders
                                (for/list ([id (in-list (cdr (syntax->list stx)))]
                                           #:when (eq? (identifier-binding id) 'lexical))
                                  (syntax-e id)))))
                 (match-lambda
                   [,pattern (binders ,@candidates)]))))
    (error 'racket-binders "the probe in the body of ~s never expanded" pattern)))

(test-case "the oracle tells a binder from a reference"
  (check-equal? (racket-binders '(app car (? number? n))) '(n))
  (check-equal? (racket-binders '(point a _)) '(a)))

(for ([pattern
       (in-list '(_ x
                    #t
                    "s"
                    1
                    #\c
                    #"b"
                    #rx"a"
                    (quote (a b))
                    'x
                    (var x)
                    (== car eq?)
                    (list a b)
                    (list a ...)
                    (list a ___)
                    (list a ..3)
                    (list a __2)
                    (list (list a b) ...)
                    (list-rest a b)
                    (list-rest a ... b)
                    (list-rest a ..2 b)
                    (list-rest a ___ b)
                    (list-rest a __2 b)
                    (list* a b)
                    (list* a ... b)
                    (list* a ..2 b)
                    (list* a ___ b)
                    (list* a __2 b)
                    (list-no-order a b)
                    (list-no-order a b ...)
                    (list-no-order a b ..2)
                    (list-no-order a b ___)
                    (list-no-order a b __2)
                    (vector a b)
                    (vector a ...)
                    (vector a ..3)
                    (vector a ___)
                    (vector a __3)
                    (hash "a" x "b" y)
                    (hash "a" x #:rest r)
                    (hash "a" x #:open)
                    (hash car x cdr (list y z))
                    (hash* ["a" x] ["b" y #:default 0])
                    (hash* ["a" x] #:closed)
                    (hash* [car x #:default cdr] #:rest r)
                    (hash-table (k v))
                    (hash-table ("a" x) (k v))
                    (hash-table (k v) ...)
                    (hash-table (k v) ___)
                    (hash-table (k v) ..2)
                    (hash-table (k v) __2)
                    (cons a b)
                    (mcons a b)
                    (box a)
                    (point a b)
                    (point a _)
                    (struct point (a b))
                    (struct point (a _))
                    (struct point _)
                    (struct* point ([a x] [b y]))
                    (struct* point ([a x]))
                    (regexp #rx"a")
                    (regexp #rx"a(b)" (list _ x))
                    (pregexp #px"a")
                    (pregexp #px"a(b)" (list whole x))
                    (and a (list b c))
                    (or (list a) (vector a))
                    (or (list a b) (vector b a))
                    (not x)
                    (not (list a b))
                    (and a (not b))
                    (app car x)
                    (app (lambda (v) (values v v)) a b)
                    (? number?)
                    (? number? n)
                    (? pair? (cons a b))
                    `(a ,b)
                    `(1 ,b ,c)
                    `(,a ...)
                    `(,a ..2)
                    `(k . ,v)
                    `#(1 ,a)
                    `#&,a
                    `#s(row ,a ,b)
                    `(1 ,@(list a b))
                    `(1 ,@(list-rest a b))
                    `(1 ,@'(x y))
                    `(x `(y ,a))
                    `(x `(y ,@(list a b)))
                    `(x `(y `(z ,a)))))])
  (test-case (format "~s binds what racket/match binds" pattern)
    (check-equal? (list->seteq (pattern-names pattern)) (list->seteq (racket-binders pattern)))))

