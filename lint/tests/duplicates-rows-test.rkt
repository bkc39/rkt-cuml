#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in "private/copies.rkt" copies))

(define (copied? one other)
  (pair? (copies `(("a.rkt" ((require racket/match
                                      "c.rkt")
                             ,one))
                   ("b.rkt" ((require racket/match
                                      "c.rkt")
                             ,other))
                   ("c.rkt" ((provide (struct-out sol)
                                      (struct-out point)
                                      (struct-out size))
                             (struct sol (coef bias))
                             (struct point (x y))
                             (struct size (x y))))))))

(for ([row
       (in-list
        '(("a struct* pattern copied verbatim is a copy" #t
                                                         (define (f s)
                                                           (match-define (struct* sol ([coef w])) s)
                                                           (* 2 w))
                                                         (define (f s)
                                                           (match-define (struct* sol ([coef w])) s)
                                                           (* 2 w)))
          ("a struct* pattern with its variables renamed is a copy"
           #t
           (define (f s)
             (match-define (struct* sol ([coef w])) s)
             (* 2 w))
           (define (g t)
             (match-define (struct* sol ([coef v])) t)
             (* 2 v)))
          ("a struct* pattern's field ids compare as written"
           #f
           (define (f s)
             (match-define (struct* sol ([coef w])) s)
             (* 2 w))
           (define (f s)
             (match-define (struct* sol ([bias w])) s)
             (* 2 w)))
          ("a for clause's #:when guard is an expression, not a binding" #f
                                                                         (define (f xs)
                                                                           (for/list ([x (in-list xs)]
                                                                                      #:when
                                                                                      (even? x))
                                                                             x))
                                                                         (define (g xs)
                                                                           (for/list ([x (in-list xs)]
                                                                                      #:when (odd? x))
                                                                             x)))
          ("a for clause's #:unless guard is an expression, not a binding"
           #f
           (define (f xss)
             (for/list ([xs (in-list xss)]
                        #:unless (null? xs))
               xs))
           (define (g xss)
             (for/list ([xs (in-list xss)]
                        #:unless (pair? xs))
               xs)))
          ("a for clause's #:break guard is an expression, not a binding"
           #f
           (define (f xs)
             (for/list ([x (in-list xs)]
                        #:break (positive? x))
               x))
           (define (g xs)
             (for/list ([x (in-list xs)]
                        #:break (negative? x))
               x)))
          ("a for clause's #:do body is an expression, not a binding"
           #f
           (define (f xs)
             (for ([x (in-list xs)]
                   #:do [(displayln x) (newline)])
               x))
           (define (g xs)
             (for ([x (in-list xs)]
                   #:do [(write x) (newline)])
               x)))
          ("a for loop with a guard, its variables renamed, is a copy" #t
                                                                       (define (evens xs)
                                                                         (for/list ([x (in-list xs)]
                                                                                    #:when (even? x))
                                                                           x))
                                                                       (define (keep ys)
                                                                         (for/list ([y (in-list ys)]
                                                                                    #:when (even? y))
                                                                           y)))
          ("an unquote inside a nested quasiquote is data" #f
                                                           (define (f a)
                                                             `(x `(y ,a)))
                                                           (define (g b)
                                                             `(x `(y ,b))))
          ("an unquote of an unquote inside a nested quasiquote is renamed" #t
                                                                            (define (f a)
                                                                              `(x `(y ,,a)))
                                                                            (define (g b)
                                                                              `(x `(y ,,b))))
          ("an unquote inside a quasiquoted box, its variable renamed, is a copy" #t
                                                                                  (define (f a)
                                                                                    `#&
                                                                                    ,a)
                                                                                  (define (g b)
                                                                                    `#&
                                                                                    ,b))
          ("an unquote inside a quasiquoted hash table, its variable renamed, is a copy"
           #t
           (define (f a)
             `#hash((k . ,a)))
           (define (g b)
             `#hash((k . ,b))))
          ("an unquote inside a quasiquoted prefab struct, its variable renamed, is a copy"
           #t
           (define (f a)
             `#s(row ,a))
           (define (g b)
             `#s(row ,b)))
          ("a legacy struct pattern's struct name is a reference, not a binding"
           #f
           (define (f v)
             (match v
               [(struct point ((? number? x) y)) y]))
           (define (g v)
             (match v
               [(struct size ((? number? x) y)) y])))
          ("a legacy struct pattern with its variables renamed is a copy"
           #t
           (define (f v)
             (match v
               [(struct point (x y)) x]))
           (define (g v)
             (match v
               [(struct point (a b)) a])))
          ("a not pattern binds nothing, so the body reads the formal" #t
                                                                       (define (f v x)
                                                                         (match v
                                                                           [(not x) x]
                                                                           [_ 0]))
                                                                       (define (g v y)
                                                                         (match v
                                                                           [(not z) y]
                                                                           [_ 0])))
          ("a regexp pattern's sub-pattern binds its names" #t
                                                            (define (f s)
                                                              (match s
                                                                [(regexp #rx"(.)" (list _ a)) a]))
                                                            (define (g s)
                                                              (match s
                                                                [(regexp #rx"(.)" (list _ b)) b])))
          ("a quasipattern binds an unquoted name inside a nested quasiquote"
           #t
           (define (f row)
             (match row
               [`(k `(,v)) v]))
           (define (g row)
             (match row
               [`(k `(,w)) w])))
          ("a repeat count is compared as written, not renamed" #f
                                                                (define (f v)
                                                                  (match v
                                                                    [(list a ..3) a]))
                                                                (define (g v)
                                                                  (match v
                                                                    [(list a ..4) a])))
          ("a hash-table pattern binds its key and value names, so a renamed copy is a copy"
           #t
           (define (f h)
             (match h
               [(hash-table (k v) ...) (map cons k v)]))
           (define (g t)
             (match t
               [(hash-table (key value) ...) (map cons key value)])))))])
  (match-define (list title copy? one other) row)
  (test-case title
    (check-equal? (copied? one other) copy?)))
