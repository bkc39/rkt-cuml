#lang racket/base

(require (only-in cuml fit score svc)
         (only-in cuml/svm svc-decision-function svc-fit svc-predict svc-result-classes)
         (only-in racket/list count)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/common-data.rkt" on-vector)
         (only-in "private/linear-data.rkt" check-law-case dtypes law-case on-matrix)
         (only-in "private/svm-data.rkt" dyadic-blobs fit-parts fresh-rows svm-case svm-fit)
         (only-in "private/svm-oracle.rkt" check-kkt))

(define-values (rows labels) (dyadic-blobs 300 4 2))
(define probe (fresh-rows 200 4 2))

(define (relabelled lo hi)
  (for/list ([v (in-list labels)])
    (if (zero? v) lo hi)))

(define encodings
  `((reals ,labels) (three-seven ,(relabelled 3 7))
                    (strings ,(relabelled "no" "yes"))
                    (symbols ,(relabelled 'no 'yes))
                    (int32 ,(on-vector labels 'int32))
                    (int64 ,(on-vector (relabelled 3 7) 'int64))
                    (float64 ,(on-vector (relabelled -1 1) 'float64))))

(define (without-classes r)
  (struct-copy svm-fit (fit-parts r) [classes #f]))

(test-gpu "L1: relabelling y leaves the fit bit for bit as it was"
  (for* ([dt (in-list dtypes)]
         [kernel (in-list '(linear rbf))])
    (define X (on-matrix rows dt))
    (define reference (without-classes (svc-fit X labels #:kernel kernel)))
    (for ([encoding (in-list encodings)])
      (match-define (list name y) encoding)
      (test-case (format "~a ~a ~a" dt kernel name)
        (law-case "L1"
          (check-equal? (without-classes (svc-fit X y #:kernel kernel)) reference))))))

(define (largest-sum ds es)
  (for/fold ([m 0.0])
            ([d (in-list ds)]
             [e (in-list es)])
    (max m (abs (+ d e)))))

(test-gpu "L2: reversing the class order negates the decision"
  (define swapped (map (lambda (v) (if (zero? v) 'yes 'no)) labels))
  (for* ([dt (in-list dtypes)]
         [kernel (in-list '(linear poly rbf))])
    (test-case (format "~a ~a" dt kernel)
      (define X (on-matrix rows dt))
      (define r (svc-fit X (relabelled 'no 'yes) #:kernel kernel))
      (define s (svc-fit X swapped #:kernel kernel))
      (law-case "L2 classes"
        (check-equal? (svc-result-classes r) '(no yes))
        (check-equal? (svc-result-classes s) '(no yes)))
      (check-kkt (svm-case "swapped" rows swapped #f 1 kernel dt 'scale) (fit-parts s))
      (define P (on-matrix probe dt))
      (define ds (device-vector->list (svc-decision-function r P)))
      (define es (device-vector->list (svc-decision-function s P)))
      (check-law-case "L2" (largest-sum ds es) 0.03)
      (law-case "L2 predictions"
        (for ([d (in-list ds)]
              [p (in-list (svc-predict r P))]
              [q (in-list (svc-predict s P))]
              #:when (> (abs d) 0.03))
          (check-equal? (list p q)
                        (if (>= d 0)
                            '(yes no)
                            '(no yes))))))))

(test-gpu "L3: score is the exact accuracy of predict, an unseen label counting as wrong"
  (define y (relabelled 'no 'yes))
  (define truth
    (for/list ([v (in-list y)]
               [i (in-naturals)])
      (if (zero? (modulo i 11)) 'maybe v)))
  (for ([dt (in-list dtypes)])
    (test-case (format "~a" dt)
      (define X (on-matrix rows dt))
      (define e (fit (svc #:kernel 'rbf) X y))
      (define predicted (svc-predict (svc-fit X y #:kernel 'rbf) X))
      (law-case "L3"
        (check-equal? (score e X truth)
                      (exact->inexact (/ (count equal? predicted truth) (length truth))))))))
