#lang racket/base

(require (only-in racket/list append-map remove-duplicates)
         (only-in "../../private/alpha.rkt" alpha-normal procedure-name))

(provide alpha-copy?
         expansion-binders
         racket-binders
         rename
         symbols)

(define oracle-namespace (make-base-namespace))

(parameterize ([current-namespace oracle-namespace])
  (namespace-require 'racket/match))

(define (shared-name? name)
  (or (eq? name 'point)
      (parameterize ([current-namespace oracle-namespace])
        (and (identifier-binding (namespace-symbol->identifier name)) #t))))

(define (symbols datum)
  (cond
    [(symbol? datum) (list datum)]
    [(pair? datum) (append (symbols (car datum)) (symbols (cdr datum)))]
    [(vector? datum) (append-map symbols (vector->list datum))]
    [(box? datum) (symbols (unbox datum))]
    [(prefab-struct-key datum) (append-map symbols (cdr (vector->list (struct->vector datum))))]
    [else '()]))

(define (rename datum
                names)
  (let walk ([one datum])
    (cond
      [(and (symbol? one) (assq one names))
       =>
       cdr]
      [(pair? one) (cons (walk (car one)) (walk (cdr one)))]
      [(vector? one) (list->vector (map walk (vector->list one)))]
      [(box? one) (box (walk (unbox one)))]
      [(prefab-struct-key one)
       =>
       (lambda (key)
         (apply make-prefab-struct key (map walk (cdr (vector->list (struct->vector one))))))]
      [else one])))

(define (expansion-binders form)
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
                 ,form)))
    (error 'expansion-binders "the probe in ~s never expanded" form)))

(define (racket-binders pattern)
  (expansion-binders `(match-lambda
                        [,pattern (binders ,@(remove-duplicates (symbols pattern)))])))

(define (free-key side)
  (lambda (name)
    (if (shared-name? name)
        name
        (format "~a@~a" name side))))

(define (alpha-copy? one other)
  (equal? (alpha-normal one (procedure-name one) (free-key "one"))
          (alpha-normal other (procedure-name other) (free-key "other"))))
