#lang racket/base

(require (only-in racket/file delete-directory/files make-directory* make-temporary-directory)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-regexp-match test-case)
         (only-in "../duplicates.rkt" copied-definitions-in describe-copies))

(define (call-with-modules modules proc)
  (define root (make-temporary-directory "duplicates-~a"))
  (dynamic-wind void
                (lambda ()
                  (for ([module (in-list modules)])
                    (match-define (list name forms) module)
                    (define file (build-path root name))
                    (make-directory* (let-values ([(directory _name _dir?) (split-path file)])
                                       directory))
                    (with-output-to-file file
                                         (lambda ()
                                           (displayln "#lang racket/base")
                                           (for-each writeln forms))))
                  (proc root))
                (lambda () (delete-directory/files root))))

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
