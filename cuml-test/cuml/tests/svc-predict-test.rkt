#lang racket/base

(require (only-in cuml/svm svc-decision-function svc-fit svc-predict svc-result-classes)
         (only-in racket/list first)
         (only-in racket/match match-define struct*)
         (only-in rackunit check-equal? test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case on-matrix)
         (only-in "private/svm-data.rkt"
                  case-fit
                  encoded-labels
                  encoding
                  fresh-rows
                  kkt-cases
                  label-encodings
                  svm-case))

(test-gpu "predict is classes[1] exactly where the decision is at least 0"
  (define c (first (kkt-cases)))
  (match-define (struct* svm-case ([X X] [y y] [dtype dt])) c)
  (define rows (fresh-rows 200 (length (first X)) 5))
  (for ([e (in-list label-encodings)])
    (match-define (struct* encoding
                           ([name name] [classes (and classes (list lo hi))] [device device]))
      e)
    (define (as-given v)
      (if (memq device '(float32 float64))
          (exact->inexact v)
          v))
    (test-case (format "~a" name)
      (define r (case-fit (struct-copy svm-case c [y (encoded-labels e y)])))
      (define decisions (device-vector->list (svc-decision-function r (on-matrix rows dt))))
      (define predicted (svc-predict r (on-matrix rows dt)))
      (law-case "R2"
        (check-equal? (if device
                          (device-vector->list predicted)
                          predicted)
                      (for/list ([d (in-list decisions)])
                        (as-given (if (>= d 0) hi lo))))
        (check-equal? (svc-result-classes r) (map as-given classes))))))

(test-gpu "a decision of exactly 0 predicts classes[1]"
  (for ([dt (in-list dtypes)])
    (test-case (format "~a" dt)
      (define r (svc-fit (on-matrix '((-1) (1) (-2) (2)) dt) '(a b a b) #:kernel 'linear))
      (define origin (on-matrix '((0)) dt))
      (law-case "R2 at 0"
        (check-equal? (device-vector->list (svc-decision-function r origin)) '(0.0))
        (check-equal? (svc-predict r origin) '(b))))))
