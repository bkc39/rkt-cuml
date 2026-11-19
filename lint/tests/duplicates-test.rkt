#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-equal? check-regexp-match test-case)
         (only-in "../duplicates.rkt" copied-definitions-in describe-copies)
         (only-in "../private/modules.rkt" with-temporary-directory write-module))

(define (call-with-modules modules proc)
  (with-temporary-directory (root "duplicates-~a")
                            (for ([module (in-list modules)])
                              (match-define (list name forms) module)
                              (write-module (build-path root name) "#lang racket/base" forms))
                            (proc root)))

(define (copies modules)
  (call-with-modules modules
                     (lambda (root)
                       (for/list ([group (in-list (copied-definitions-in (list root)))])
                         (for/list ([line (in-list (cdr (regexp-split #rx"\n  "
                                                                      (describe-copies group))))])
                           (regexp-replace #rx"^.*/" line ""))))))

(define helper
  '(define (mean xs)
     (/ (apply + xs) (length xs))))

(define renamed-helper
  '(define (average values)
     (/ (apply + values) (length values))))

(match-define (list worst-gap largest-gap)
  '((define (worst xs ys)
      (for/fold ([most 0.0])
                ([x (in-list xs)]
                 [y (in-list ys)])
        (let ([gap (abs (- x y))]) (max most gap))))
    (define (largest-gap as bs)
      (for/fold ([acc 0.0])
                ([a (in-list as)]
                 [b (in-list bs)])
        (let ([d (abs (- a b))]) (max acc d))))))

(test-case "a helper copied into a second file is refused at both sites"
  (check-equal? (copies `(("a-test.rkt" (,helper)) ("private/b-data.rkt" (,helper))))
                '(("a-test.rkt:2: mean" "b-data.rkt:2: mean"))))

(test-case "a copy is found after renaming the helper, its formals and its local bindings"
  (check-equal? (copies `(("a-test.rkt" (,helper)) ("b-test.rkt" (,renamed-helper))))
                '(("a-test.rkt:2: mean" "b-test.rkt:2: average")))
  (check-equal? (copies `(("a-test.rkt" (,worst-gap)) ("b-test.rkt" (,largest-gap))))
                '(("a-test.rkt:2: worst" "b-test.rkt:2: largest-gap"))))

(test-case "the refusal names what to change"
  (call-with-modules
   `(("a-test.rkt" (,helper)) ("b-test.rkt" (,helper)))
   (lambda (root)
     (check-regexp-match
      #rx"keep one, in the family's shared module [(]cuml-test's private/common-data[.]rkt when two families use it[)], and require it from there"
      (describe-copies (car (copied-definitions-in (list root))))))))

(test-case "one definition per helper, or two in one file, passes"
  (check-equal? (copies `(("a-test.rkt" (,helper))
                          ("b-test.rkt" ((require (only-in "a-test.rkt" mean)) (mean '(1 2))))))
                '())
  (check-equal? (copies `(("a-test.rkt" (,helper ,renamed-helper)))) '()))

(test-case "bodies that differ in a free name, or in a module's own definitions, are not copies"
  (check-equal? (copies `(("a-test.rkt" (,helper)) ("b-test.rkt" ((define (mean xs)
                                                                    (/ (apply * xs) (length xs)))))))
                '())
  (check-equal? (copies '(("a-test.rkt" ((define twin "a.py") (define (results r)
                                                                (run twin r))))
                          ("b-test.rkt" ((define twin "b.py") (define (results r)
                                                                (run twin r))))))
                '())
  (check-equal? (copies '(("a-test.rkt" ((struct fixture (name)) (define (key f)
                                                                   (fixture-name f))))
                          ("b-test.rkt" ((struct fixture (name)) (define (key f)
                                                                   (fixture-name f))))))
                '()))

(test-case "a table passed to define/who is a reference, not a formal"
  (check-equal? (copies '(("a.rkt" ((define/who (a-fit who X #:hyperparameters a-params)
                                      (fitted who X (bound-params a-params)))))
                          ("b.rkt" ((define/who (b-fit who X #:hyperparameters b-params)
                                      (fitted who X (bound-params b-params)))))))
                '()))

(test-case "every procedure-valued definition form is compared"
  (for ([form (in-list '((define ((near x) y)
                           (< (abs (- x y)) 1e-6))
                         (define pick
                           (case-lambda
                             [(x) x]
                             [(x y) (+ x y)]))
                         (define past-sum (const (exact->inexact (expt 2 125))))
                         (define add-one (curry + 1))
                         (define/contract (double x)
                           (-> real? real?)
                           (* 2 x))))])
    (check-equal? (length (copies `(("a.rkt" ((require racket/contract
                                                       racket/function)
                                              ,form))
                                    ("b.rkt" ((require racket/contract
                                                       racket/function)
                                              ,form)))))
                  1
                  (format "~s" form))))

(test-case "a definition inside module+ or module* is compared"
  (check-equal? (copies `(("a.rkt" ((module+ test
                                      ,helper)))
                          ("b.rkt" ((module* main #f
                                      ,helper)))))
                '(("a.rkt:2: mean" "b.rkt:2: mean"))))

(test-case "a constant is per-file data, not a copy"
  (check-equal? (copies '(("a.rkt" ((define tolerance (* 64 (expt 2.0 -53)))))
                          ("b.rkt" ((define tolerance (* 64 (expt 2.0 -53)))))))
                '()))

(test-case "free names compare by the binding they import, not by their spelling"
  (define first-of
    '(define (head xs)
       (first xs)))
  (define elsewhere
    '("c.rkt" ((provide first) (define (first xs)
                                 (cadr xs)))))
  (check-equal? (copies `(("a.rkt" ((require (only-in racket/list first)) ,first-of))
                          ("b.rkt" ((require (only-in "c.rkt" first)) ,first-of))
                          ,elsewhere))
                '())
  (check-equal? (copies `(("a.rkt" ((require (only-in "c.rkt" first)) ,first-of))
                          ("b.rkt" ((require (only-in "c.rkt" first)) ,first-of))
                          ,elsewhere))
                '(("a.rkt:3: head" "b.rkt:3: head")))
  (check-equal? (copies `(("a.rkt" ((require (only-in racket/list first)) ,first-of))
                          ("b.rkt" ((require (prefix-in list: racket/list)) (define (head xs)
                                                                              (list:first xs))))))
                '(("a.rkt:3: head" "b.rkt:3: head"))))

(test-case "a name a nested binding shadows is renamed only inside it"
  (check-equal? (copies '(("a.rkt" ((require (only-in "c.rkt" x z)) (define (f y)
                                                                      (+ (let ([x 1]) x) x))))
                          ("b.rkt" ((require (only-in "c.rkt" x z)) (define (f y)
                                                                      (+ (let ([z 1]) z) z))))
                          ("c.rkt" ((provide x
                                             z)
                                    (define x 10)
                                    (define z 20)))))
                '()))

(test-case "quoted data is compared as written, never renamed"
  (check-equal? (copies '(("a.rkt" ((define (tag x)
                                      (list 'x x))))
                          ("b.rkt" ((define (tag y)
                                      (list 'y y))))))
                '())
  (check-equal? (copies '(("a.rkt" ((define (tag x)
                                      `(x ,x))))
                          ("b.rkt" ((define (tag y)
                                      `(x ,y))))))
                '(("a.rkt:2: tag" "b.rkt:2: tag"))))

(test-case "a quasiquoted match pattern binds only its unquoted names, so its literals are data"
  (check-equal? (copies '(("a.rkt" ((require (only-in racket/match match)) (define (tag row)
                                                                             (match row
                                                                               [`(x ,v) v]))))
                          ("b.rkt" ((require (only-in racket/match match)) (define (tag row)
                                                                             (match row
                                                                               [`(y ,v) v]))))))
                '())
  (check-equal? (copies '(("a.rkt" ((require (only-in racket/match match)) (define (tag row)
                                                                             (match row
                                                                               [`(x ,v) v]))))
                          ("b.rkt" ((require (only-in racket/match match)) (define (tag row)
                                                                             (match row
                                                                               [`(x ,w) w]))))))
                '(("a.rkt:3: tag" "b.rkt:3: tag"))))

(test-case "a definition inside a form the lint does not know is renamed over that form"
  (check-equal? (copies '(("a.rkt" ((require (only-in "c.rkt" with-scratch))
                                    (define (f xs)
                                      (with-scratch "x" (define total (apply + xs)) total))))
                          ("b.rkt" ((require (only-in "c.rkt" with-scratch))
                                    (define (f ys)
                                      (with-scratch "x" (define sum (apply + ys)) sum))))
                          ("c.rkt" ((provide with-scratch) (define with-scratch list)))))
                '(("a.rkt:3: f" "b.rkt:3: f"))))

(for ([binder (in-list '(cond
                          when
                          unless
                          parameterize))]
      [pair (in-list '(((define (f xs)
                          (cond
                            [(pair? xs)
                             (define total (apply + xs))
                             (/ total (length xs))]
                            [else 0]))
                        (define (g ys)
                          (cond
                            [(pair? ys)
                             (define sum (apply + ys))
                             (/ sum (length ys))]
                            [else 0])))
                       ((define (f xs)
                          (when (pair? xs)
                            (define total (apply + xs))
                            (/ total (length xs))))
                        (define (g ys)
                          (when (pair? ys)
                            (define sum (apply + ys))
                            (/ sum (length ys)))))
                       ((define (f xs)
                          (unless (null? xs)
                            (define total (apply + xs))
                            (/ total (length xs))))
                        (define (g ys)
                          (unless (null? ys)
                            (define sum (apply + ys))
                            (/ sum (length ys)))))
                       ((define (f xs)
                          (parameterize ([current-output-port (current-error-port)])
                            (define total (apply + xs))
                            (/ total (length xs))))
                        (define (g ys)
                          (parameterize ([current-output-port (current-error-port)])
                            (define sum (apply + ys))
                            (/ sum (length ys)))))))])
  (test-case (format "a definition inside ~a is renamed by scope" binder)
    (match-define (list one other) pair)
    (check-equal? (copies `(("a.rkt" (,one)) ("b.rkt" (,other)))) '(("a.rkt:2: f" "b.rkt:2: g")))))

(for ([scope (in-list '((when ((define (f x)
                                 (when (positive? x)
                                   (define x 5)
                                   x)))
                          ((define (f y)
                             (when (positive? y)
                               (define z 5)
                               z)))
                          ((define limit -1) (define (f x)
                                               (when (positive? limit)
                                                 (define limit 5)
                                                 limit))))
                        (cond
                          [(define (f x)
                             (cond
                               [(positive? x)
                                (define x 5)
                                x]
                               [else 0]))]
                          [(define (f y)
                             (cond
                               [(positive? y)
                                (define z 5)
                                z]
                               [else 0]))]
                          [(define limit -1)
                           (define (f x)
                             (cond
                               [(positive? limit)
                                (define limit 5)
                                limit]
                               [else 0]))])
                        (parameterize ([define
                                        (f x)
                                        (parameterize ([current-command-line-arguments x])
                                          (define x (vector))
                                          x)])
                          ((define (f y)
                             (parameterize ([current-command-line-arguments y])
                               (define z (vector))
                               z)))
                          ((define limit (vector "x"))
                           (define (f x)
                             (parameterize ([current-command-line-arguments limit])
                               (define limit (vector))
                               limit))))
                        (unless ((define (f x)
                                   (unless (zero? x)
                                     (define x 5)
                                     x)))
                          ((define (f y)
                             (unless (zero? y)
                               (define z 5)
                               z)))
                          ((define limit -1) (define (f x)
                                               (unless (zero? limit)
                                                 (define limit 5)
                                                 limit))))
                        (case ((define (f x)
                                 (case x
                                   [(x)
                                    (define x 5)
                                    x]
                                   [else 0])))
                          [(define (f y)
                             (case y
                               [(x)
                                (define z 5)
                                z]
                               [else 0]))]
                          [(define (f x)
                             (case x
                               [(limit)
                                (define limit 5)
                                limit]
                               [else 0]))])
                        (with-handlers ([define
                                         (f x)
                                         (with-handlers ([x void])
                                           (define x 5)
                                           x)])
                          ((define (f y)
                             (with-handlers ([y void])
                               (define z 5)
                               z)))
                          ((define limit -1) (define (f x)
                                               (with-handlers ([limit void])
                                                 (define limit 5)
                                                 limit))))))])
  (match-define (list binder one copy other) scope)
  (test-case (format "a definition inside ~a shadows only its body, so a renamed copy matches" binder)
    (check-equal? (copies `(("a.rkt" ,one) ("b.rkt" ,copy))) '(("a.rkt:2: f" "b.rkt:2: f"))))
  (test-case (format "a definition inside ~a is not bound over its test or bindings" binder)
    (check-equal? (copies `(("a.rkt" ,one) ("b.rkt" ,other))) '())))

(define (copied? one other)
  (pair? (copies `(("a.rkt" ((require racket/match "c.rkt") ,one))
                   ("b.rkt" ((require racket/match "c.rkt") ,other))
                   ("c.rkt" ((provide (struct-out sol) (struct-out point) (struct-out size))
                             (struct sol (coef bias))
                             (struct point (x y))
                             (struct size (x y))))))))

(for ([row (in-list '(("a struct* pattern copied verbatim is a copy"
                       #t
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
                      ("a for clause's #:when guard is an expression, not a binding"
                       #f
                       (define (f xs)
                         (for/list ([x (in-list xs)] #:when (even? x)) x))
                       (define (g xs)
                         (for/list ([x (in-list xs)] #:when (odd? x)) x)))
                      ("a for clause's #:unless guard is an expression, not a binding"
                       #f
                       (define (f xss)
                         (for/list ([xs (in-list xss)] #:unless (null? xs)) xs))
                       (define (g xss)
                         (for/list ([xs (in-list xss)] #:unless (pair? xs)) xs)))
                      ("a for clause's #:break guard is an expression, not a binding"
                       #f
                       (define (f xs)
                         (for/list ([x (in-list xs)] #:break (positive? x)) x))
                       (define (g xs)
                         (for/list ([x (in-list xs)] #:break (negative? x)) x)))
                      ("a for clause's #:do body is an expression, not a binding"
                       #f
                       (define (f xs)
                         (for ([x (in-list xs)] #:do [(displayln x) (newline)]) x))
                       (define (g xs)
                         (for ([x (in-list xs)] #:do [(write x) (newline)]) x)))
                      ("a for loop with a guard, its variables renamed, is a copy"
                       #t
                       (define (evens xs)
                         (for/list ([x (in-list xs)] #:when (even? x)) x))
                       (define (keep ys)
                         (for/list ([y (in-list ys)] #:when (even? y)) y)))
                      ("an unquote inside a nested quasiquote is data"
                       #f
                       (define (f a)
                         `(x `(y ,a)))
                       (define (g b)
                         `(x `(y ,b))))
                      ("an unquote of an unquote inside a nested quasiquote is renamed"
                       #t
                       (define (f a)
                         `(x `(y ,,a)))
                       (define (g b)
                         `(x `(y ,,b))))
                      ("an unquote inside a quasiquoted box, its variable renamed, is a copy"
                       #t
                       (define (f a)
                         `#&,a)
                       (define (g b)
                         `#&,b))
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
                         (match v [(struct point (x y)) x]))
                       (define (g v)
                         (match v [(struct size (x y)) x])))
                      ("a legacy struct pattern with its variables renamed is a copy"
                       #t
                       (define (f v)
                         (match v [(struct point (x y)) x]))
                       (define (g v)
                         (match v [(struct point (a b)) a])))
                      ("a not pattern binds nothing, so the body reads the formal"
                       #t
                       (define (f v x)
                         (match v [(not x) x] [_ 0]))
                       (define (g v y)
                         (match v [(not z) y] [_ 0])))
                      ("a regexp pattern's sub-pattern binds its names"
                       #t
                       (define (f s)
                         (match s [(regexp #rx"(.)" (list _ a)) a]))
                       (define (g s)
                         (match s [(regexp #rx"(.)" (list _ b)) b])))
                      ("a quasipattern binds an unquoted name inside a nested quasiquote"
                       #t
                       (define (f row)
                         (match row [`(k `(,v)) v]))
                       (define (g row)
                         (match row [`(k `(,w)) w])))))])
  (match-define (list title copy? one other) row)
  (test-case title
    (check-equal? (copied? one other) copy?)))
