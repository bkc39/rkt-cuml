#lang racket/base

(require (only-in racket/list append-map argmin remove-duplicates)
         (only-in racket/match match match-define match-lambda)
         (only-in rackunit check-equal? test-case)
         (only-in "private/oracle.rkt" alpha-copy? racket-value rename symbols))

(struct node (pattern value))

(define names '(k j n))

(define seeds
  (cons 20261119
        (match (getenv "CUML_LINT_FUZZ_SEED")
          [#f '()]
          [seed (list (string->number seed))])))

(define wanted
  (match (getenv "CUML_LINT_FUZZ_COUNT")
    [#f 400]
    [count (string->number count)]))

(define (pick choices)
  (list-ref choices (random (length choices))))

(define (chance p)
  (< (random) p))

(define (repeat-minimum repeat)
  (match repeat
    [(or '..1 '__1) 1]
    ['..2 2]
    [_ 0]))

(define (repeated value repeat)
  (for/list ([_ (in-range (+ (repeat-minimum repeat) (random 3)))])
    (value)))

(define (leaf-value)
  (random 3))

(define (name-node name)
  (node name leaf-value))

(define wildcard (node '_ leaf-value))

(define (literal-node)
  (define literal (pick '(1 'x "s")))
  (node literal (lambda () literal)))

(define (leaf)
  (cond
    [(chance 0.6) (name-node (pick names))]
    [(chance 0.15) (node `(var ,(pick names)) leaf-value)]
    [(chance 0.5) wildcard]
    [else (literal-node)]))

(define (random-repeat)
  (pick '(... ... ___ ..1 ..2 ..0 __1)))

(define (items depth count #:repeats? [repeats? #t])
  (for/list ([_ (in-range count)])
    (cons (if (and (= depth 0) (chance 0.3))
              (if (chance 0.5) wildcard (leaf))
              (generate (sub1 depth)))
          (and repeats? (chance 0.3) (random-repeat)))))

(define (items-pattern items)
  (append-map (match-lambda
                [(cons one #f) (list (node-pattern one))]
                [(cons one repeat) (list (node-pattern one) repeat)])
              items))

(define (items-values items)
  (append-map (match-lambda
                [(cons one #f) (list ((node-value one)))]
                [(cons one repeat) (repeated (node-value one) repeat)])
              items))

(define (list-node depth)
  (define elements (items depth (random 4)))
  (node `(list ,@(items-pattern elements)) (lambda () `(list ,@(items-values elements)))))

(define (list-rest-node depth)
  (define elements (items depth (random 3)))
  (define tail (generate (sub1 depth)))
  (node `(,(pick '(list-rest list*)) ,@(items-pattern elements) ,(node-pattern tail))
        (lambda () `(list* ,@(items-values elements) ,((node-value tail))))))

(define (vector-node depth)
  (define elements
    (if (chance 0.3)
        (append (items depth (random 2) #:repeats? #f)
                (list (cons wildcard (random-repeat)))
                (items depth (random 2) #:repeats? #f))
        (items depth (random 4))))
  (node `(vector ,@(items-pattern elements)) (lambda () `(vector ,@(items-values elements)))))

(define (list-no-order-node depth)
  (define elements (items depth (add1 (random 3)) #:repeats? #f))
  (define repeat (and (chance 0.4) (cons (generate (sub1 depth)) (random-repeat))))
  (define all (append elements (if repeat (list repeat) '())))
  (node `(list-no-order ,@(items-pattern all)) (lambda () `(list ,@(items-values all)))))

(define (key-node depth)
  (if (chance 0.5)
      (let ([key (pick '('x 'y 7 "s"))]) (node key (lambda () key)))
      (generate (sub1 depth))))

(define (hash-table-node depth)
  (define entries
    (for/list ([_ (in-range (add1 (random 2)))])
      (cons (key-node depth) (generate (sub1 depth)))))
  (node `(hash-table ,@(for/list ([entry (in-list entries)])
                         (list (node-pattern (car entry)) (node-pattern (cdr entry))))
                     ,@(if (chance 0.25) '(...) '()))
        (lambda ()
          `(hash ,@(append-map (lambda (entry) (list ((node-value (car entry))) ((node-value (cdr entry)))))
                               entries)))))

(define (spine-tail depth)
  (match (random 3)
    [0 (node (pick names) (lambda () `(list ,@(repeated leaf-value '..0))))]
    [1 (node `(var ,(pick names)) (lambda () `(list ,@(repeated leaf-value '..0))))]
    [_ (list-node (sub1 depth))]))

(define (spine-node depth)
  (define head (generate (sub1 depth)))
  (define tail (spine-tail depth))
  (match (random 3)
    [0 (node `(cons ,(node-pattern head) ,(node-pattern tail))
             (lambda () `(cons ,((node-value head)) ,((node-value tail)))))]
    [1 (node `(,(pick '(list-rest list*)) ,(node-pattern head) ,(node-pattern tail))
             (lambda () `(list* ,((node-value head)) ,((node-value tail)))))]
    [_ (quasi-node (sub1 depth) #:tail (and (chance 0.5) (spine-tail depth)))]))

(define (element depth)
  (match (random 9)
    [0 (define one (generate (sub1 depth)))
       (node (list (list 'unquote (node-pattern one))) (lambda () `(list ,((node-value one)))))]
    [1 (define one (generate (sub1 depth)))
       (define repeat (random-repeat))
       (node (list (list 'unquote (node-pattern one)) repeat)
             (lambda () `(list ,@(repeated (node-value one) repeat))))]
    [2 (define name (pick names))
       (node (list (list 'unquote-splicing name)) (lambda () `(list ,@(repeated leaf-value '..0))))]
    [3 (define one (list-node (sub1 depth)))
       (node (list (list 'unquote-splicing (node-pattern one))) (node-value one))]
    [4 (define one (spine-node depth))
       (node (list (list 'unquote-splicing (node-pattern one))) (node-value one))]
    [5 (node '(x) (lambda () ''(x)))]
    [6 (define one (template (sub1 depth)))
       (node (list (node-pattern one)) (lambda () `(list ,((node-value one)))))]
    [_ (define one (template (sub1 depth) #:tail #f))
       (node (list (list->vector (node-pattern one)))
             (lambda () `(list (list->vector ,((node-value one))))))]))

(define (template depth #:tail [tail (and (chance 0.2) (generate (sub1 depth)))])
  (define elements
    (for/list ([_ (in-range (random 4))])
      (element depth)))
  (define front (append-map node-pattern elements))
  (node (if tail
            (append front (list 'unquote (node-pattern tail)))
            front)
        (lambda ()
          `(append ,@(for/list ([one (in-list elements)])
                       ((node-value one)))
                   ,@(if tail
                         (list ((node-value tail)))
                         '())))))

(define (quasi-node depth #:tail [tail (and (chance 0.2) (generate (sub1 depth)))])
  (define body (template depth #:tail tail))
  (node (list 'quasiquote (node-pattern body)) (node-value body)))

(define (wrap-node depth)
  (define one (generate (sub1 depth)))
  (define other (generate (sub1 depth)))
  (define (value) ((node-value one)))
  (define (other-value) ((node-value other)))
  (define pattern (node-pattern one))
  (match (random 11)
    [0 (node `(cons ,pattern ,(node-pattern other)) (lambda () `(cons ,(value) ,(other-value))))]
    [1 (node `(box ,pattern) (lambda () `(box ,(value))))]
    [2 (node `(point ,pattern ,(node-pattern other)) (lambda () `(point ,(value) ,(other-value))))]
    [3
     (node `(struct point (,pattern ,(node-pattern other)))
           (lambda () `(point ,(value) ,(other-value))))]
    [4 (node `(struct* point ([b ,pattern])) (lambda () `(point ,(other-value) ,(value))))]
    [5 (node `(? values ,pattern) value)]
    [6 (node `(app values ,pattern) value)]
    [7
     (define name (pick names))
     (node (if (chance 0.5)
               `(and ,name ,pattern)
               `(and ,pattern ,name))
           value)]
    [8
     (define wrapper (pick '(box list vector)))
     (node (if (chance 0.5)
               `(or ,pattern (,wrapper ,pattern))
               `(or (,wrapper ,pattern) ,pattern))
           (lambda ()
             (if (chance 0.5)
                 (value)
                 `(,wrapper ,(value)))))]
    [9 (node `(app box (box ,pattern)) value)]
    [_ (node `(? values ,(pick names) ,pattern) value)]))

(define (generate depth)
  (if (<= depth 0)
      (leaf)
      (match (random 9)
        [0 (leaf)]
        [1 (list-node depth)]
        [2 (list-rest-node depth)]
        [3 (vector-node depth)]
        [4 (list-no-order-node depth)]
        [5 (hash-table-node depth)]
        [6 (quasi-node depth)]
        [_ (wrap-node depth)])))

(define (size datum)
  (cond
    [(pair? datum) (+ (size (car datum)) (size (cdr datum)))]
    [(vector? datum) (add1 (size (vector->list datum)))]
    [(null? datum) 0]
    [else 1]))

(define (racket-answers pattern pairs samples)
  (with-handlers ([exn:fail? (lambda (_) #f)])
    (racket-value
     `(let ()
        (struct point (a b) #:transparent)
        (define (first-match body guard)
          (for/or ([v (in-list (list ,@samples))])
            (define answer
              (with-handlers ([exn:fail? (lambda (_) 'unmatched)])
                (body 'formal v)))
            (and (not (eq? answer 'unmatched)) (list answer (guard 'formal v)))))
        (list ,@(for/list ([pair (in-list pairs)])
                  (match-define (cons failure read) pair)
                  `(first-match (lambda (,read v)
                                  (match v
                                    [,pattern
                                     (=> ,failure)
                                     (cond
                                       [(procedure? ,read) 'failure]
                                       [(eq? ,read 'formal) 'formal]
                                       [else 'pattern])]
                                    [_ 'unmatched]))
                                (lambda (,read v)
                                  (match v
                                    [,pattern (=> ,failure) #:when (procedure? ,read) 'failure]
                                    [,pattern (=> ,failure) #:when (eq? ,read 'formal) 'formal]
                                    [,pattern 'pattern]
                                    [_ 'unmatched])))))))))

(define (lint-answer clause pattern failure read)
  (define (procedure formal pattern failure read)
    `(define (g ,formal v)
       (match v
         ,(clause pattern failure read))))
  (define one (procedure read pattern failure read))
  (define copies
    (for/list ([kind (in-list '(formal failure pattern))]
               [other (in-list (list (procedure 'z pattern failure 'z)
                                     (procedure read pattern 'z 'z)
                                     (procedure read (rename pattern (list (cons read 'z))) failure 'z)))]
               #:when (alpha-copy? one other))
      kind))
  (match copies
    [(list kind) kind]
    [_ (cons 'inconsistent copies)]))

(define (lint-answers pattern failure read)
  (for/list ([clause (in-list (list (lambda (pattern failure read) `[,pattern (=> ,failure) ,read])
                                    (lambda (pattern failure read)
                                      `[,pattern (=> ,failure) #:when ,read 0])))])
    (lint-answer clause pattern failure read)))

(define (row-disagreements pattern pair answer)
  (match-define (cons failure read) pair)
  (for/list ([position (in-list '(body when))]
             [racket (in-list answer)]
             [lint (in-list (lint-answers pattern failure read))]
             #:unless (equal? racket lint))
    (list (list position (if (eq? failure read) 'failure-id 'other-name) racket lint) pattern pair)))

(define (disagreements seed)
  (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
    (random-seed seed)
    (let loop ([found '()]
               [patterns 0])
      (cond
        [(< patterns wanted)
         (define one (generate (add1 (random 3))))
         (define pattern (node-pattern one))
         (define bound (remove-duplicates (filter (lambda (name) (memq name names)) (symbols pattern))))
         (define pairs
           (for*/list ([failure (in-list bound)]
                       [read (in-list bound)])
             (cons failure read)))
         (define samples
           (for/list ([_ (in-range 30)])
             ((node-value one))))
         (define answers (or (and (pair? pairs) (racket-answers pattern pairs samples)) '()))
         (define rows
           (for/list ([pair (in-list pairs)]
                      [answer (in-list answers)]
                      #:when answer)
             (row-disagreements pattern pair answer)))
         (if (null? rows)
             (loop found patterns)
             (loop (apply append found rows) (add1 patterns)))]
        [else found]))))

(define (smallest-by-kind found)
  (for/list ([kind (in-list (remove-duplicates (map car found)))])
    (define same (filter (lambda (one) (equal? (car one) kind)) found))
    (match (argmin (match-lambda
                     [(list _ pattern _) (size pattern)])
                   same)
      [(list _ pattern (cons failure read))
       (list kind (length same) 'smallest pattern 'failure-id failure 'read read)])))

(for ([seed (in-list seeds)])
  (test-case (format "the copy check scopes a failure id as Racket does on ~a fuzzed patterns, seed ~a"
                     wanted
                     seed)
    (check-equal? (smallest-by-kind (disagreements seed)) '())))
