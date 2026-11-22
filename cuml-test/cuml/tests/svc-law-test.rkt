#lang racket/base

(require (only-in racket/match match-define struct*)
         (only-in rackunit check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" check-law-case law-case)
         (only-in "private/svm-data.rkt" case-fit fit-parts kkt-cases svm-case svm-case-name)
         (only-in "private/svm-oracle.rkt" kkt kkt-bounds kkt-of))

(define tol 1e-3)

(define (check-kkt c)
  (match-define (struct* svm-case ([dtype dt])) c)
  (define-values (box balance gap-bound) (kkt-bounds dt tol))
  (match-define (struct* kkt ([signs-agree signs]
                              [box-excess excess]
                              [balance ratio]
                              [gap gap]
                              [violation violation]
                              [support-exact support]
                              [counts counts]))
    (kkt-of c (fit-parts (case-fit c))))
  (law-case "K1 signs"
    (check-true signs))
  (check-law-case "K1 box" excess box)
  (check-law-case "K2" ratio balance)
  (check-law-case "K3" gap gap-bound)
  (check-law-case "K4" violation gap-bound)
  (law-case "K6"
    (check-true support)
    (check-true counts)))

(test-gpu "K: the KKT law on the grid"
  (for ([c (in-list (kkt-cases))])
    (test-case (svm-case-name c)
      (check-kkt c))))
