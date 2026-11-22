#lang racket/base

(require (only-in racket/list first)
         (only-in racket/match match-define struct*)
         (only-in rackunit check-equal? test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/svm-data.rkt" case-fit kkt-cases svc-result->svm-fit svm-fit)
         (only-in "private/svm-oracle.rkt" kkt-of kkt-rows-exact kkt-support-exact))

(define (exactness c fit)
  (define k (kkt-of c fit))
  (list (kkt-support-exact k) (kkt-rows-exact k)))

(test-gpu "K6 and K7 refuse a repeated support index and a support vector off X's rows"
  (define c (first (kkt-cases)))
  (define fit (svc-result->svm-fit (case-fit c)))
  (match-define (struct* svm-fit ([sv (cons (cons v row) rows)] [support (list* i _ more)])) fit)
  (test-case "the fit as returned"
    (check-equal? (exactness c fit) '(#t #t)))
  (test-case "a repeated index"
    (check-equal? (exactness c (struct-copy svm-fit fit [support (list* i i more)])) '(#f #f)))
  (test-case "a support vector off its row"
    (check-equal? (exactness c (struct-copy svm-fit fit [sv (cons (cons (+ v 1/64) row) rows)]))
                  '(#t #f))))
