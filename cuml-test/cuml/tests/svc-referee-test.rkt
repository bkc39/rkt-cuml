#lang racket/base

(require (only-in cuml/svm svc-decision-function svc-predict svc-result-classes)
         (only-in racket/match match-define struct*)
         (only-in rackunit check-equal? test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/common-data.rkt" on-vector)
         (only-in "private/linear-data.rkt" check-law-case dtypes law-case on-matrix)
         (only-in "private/svm-data.rkt"
                  case-fit
                  fit-parts
                  fresh-rows
                  kkt-cases
                  svm-case
                  svm-case-name)
         (only-in "private/svm-oracle.rkt" decision-bound exact-decision))

(define fresh 200)

(define (worst-ratio fit dt rows decisions)
  (for/fold ([worst 0])
            ([x (in-list rows)]
             [d (in-list decisions)])
    (define gap (abs (- (inexact->exact d) (exact-decision fit x))))
    (define bound (decision-bound fit dt x))
    (max worst
         (if (zero? bound)
             (if (zero? gap) 0 +inf.0)
             (/ gap bound)))))

(test-gpu "R1: the decision is within B of the exact decision"
  (for ([c (in-list (kkt-cases))])
    (match-define (struct* svm-case ([X X] [dtype dt])) c)
    (test-case (svm-case-name c)
      (define r (case-fit c))
      (define rows (fresh-rows fresh (length (car X)) (length X)))
      (define decisions (device-vector->list (svc-decision-function r (on-matrix rows dt))))
      (check-law-case "R1" (worst-ratio (fit-parts r) dt rows decisions) 1))))

(define label-sets
  `((reals (0 1) #f) (named ("no" "yes") #f) (symbols (no yes) #f) (int32 (0 1) int32)
                     (int64 (3 7) int64) (float32 (0 1) float32) (float64 (-1 1) float64)))

(test-gpu "R2: predict is classes[1] exactly where the decision is at least 0"
  (define c (car (kkt-cases)))
  (match-define (struct* svm-case ([X X] [y y] [dtype dt])) c)
  (for ([set (in-list label-sets)])
    (match-define (list name classes device) set)
    (match-define (list lo hi) classes)
    (define labels
      (for/list ([v (in-list y)])
        (if (zero? v) lo hi)))
    (define (as-given v)
      (if (memq device '(float32 float64))
          (exact->inexact v)
          v))
    (test-case (format "~a" name)
      (define r
        (case-fit (struct-copy svm-case c [y (if device (on-vector labels device) labels)])))
      (define rows (fresh-rows fresh (length (car X)) 5))
      (define decisions (device-vector->list (svc-decision-function r (on-matrix rows dt))))
      (define predicted (svc-predict r (on-matrix rows dt)))
      (law-case "R2"
        (check-equal? (if device
                          (device-vector->list predicted)
                          predicted)
                      (for/list ([d (in-list decisions)])
                        (as-given (if (>= d 0) hi lo))))
        (check-equal? (svc-result-classes r) (map as-given classes))))))
