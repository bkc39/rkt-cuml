#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in "private/logistic-sklearn-twin.rkt" collapsed))

(define rows
  '([(a a b c c c) (2 1 3) (a b c)] [(a b b) (1 0 2) (a b)] [(a) (0 1 0) (a)] [() (0 0) ()]))

(test-case "collapsed keeps one label per repeated row and none for a zero count"
  (for ([row (in-list rows)])
    (match-define (list labels counts want) row)
    (check-equal? (collapsed labels counts) want (format "~a" row))))
