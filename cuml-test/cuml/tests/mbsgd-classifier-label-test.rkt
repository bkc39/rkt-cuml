#lang racket/base

(require (only-in cuml/linear-model
                  logistic-regression-fit
                  logistic-regression-result-label-kind
                  mbsgd-classifier-decision-function
                  mbsgd-classifier-fit
                  mbsgd-classifier-predict
                  mbsgd-classifier-result
                  mbsgd-classifier-result-classes
                  mbsgd-classifier-result-coef
                  mbsgd-classifier-result-label-kind)
         (only-in racket/list make-list)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-vector->list device-vector? dtype list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case on-matrix on-vector)
         (only-in "private/mbsgd-data.rkt" blob-rows clf-parts)
         (only-in "private/refusal-rows.rkt" check-row row row-id))

(define-values (blobs codes) (blob-rows 120 3 31))
(define losses '(hinge log squared-loss))

(define (relabel table)
  (map (lambda (c) (vector-ref table c)) codes))

(define (host v)
  (if (device-vector? v)
      (device-vector->list v)
      v))

(define (kinds)
  (list (list "int32 0/1" (on-vector codes 'int32) '(0 1) 'int32)
        (list "int64 3/7" (on-vector (relabel #(3 7)) 'int64) '(3 7) 'int64)
        (list "float64 0.0/1.0" (on-vector codes 'float64) '(0.0 1.0) 'float64)
        (list "host 3/7" (relabel #(3 7)) '(3 7) 'list)
        (list "strings" (relabel #("B" "M")) '("B" "M") 'list)
        (list "symbols" (relabel #(a b)) '(a b) 'list)))

(test-gpu "every kind of two labels gives one fit, its own sorted classes, and predicts in its kind"
  (for* ([dt (in-list dtypes)]
         [loss (in-list losses)])
    (define X (on-matrix blobs dt))
    (define reference (clf-parts (mbsgd-classifier-fit X (on-vector codes 'int32) #:loss loss)))
    (for ([k (in-list (kinds))])
      (law-case (format "~a ~a ~a" dt loss (car k))
        (define r (mbsgd-classifier-fit X (cadr k) #:loss loss))
        (check-equal? (clf-parts r) reference)
        (check-equal? (mbsgd-classifier-result-classes r) (caddr k))
        (define predicted (mbsgd-classifier-predict r X))
        (if (eq? (cadddr k) 'list)
            (check-true (list? predicted))
            (check-equal? (dtype predicted) (cadddr k)))
        (check-true (andmap (lambda (l) (and (member l (caddr k)) #t)) (host predicted)))))))

(test-gpu "swapping which class sorts first negates the hinge fit"
  (for* ([dt (in-list dtypes)]
         [intercept? (in-list '(#t #f))])
    (law-case (format "~a fit-intercept ~a" dt intercept?)
      (define X (on-matrix blobs dt))
      (define r (clf-parts (mbsgd-classifier-fit X (relabel #(a b)) #:fit-intercept intercept?)))
      (define swapped
        (clf-parts (mbsgd-classifier-fit X (relabel #(b a)) #:fit-intercept intercept?)))
      (check-equal? swapped (list (map - (car r)) (- 0.0 (cadr r)))))))

(define (threshold loss)
  (if (eq? loss 'squared-loss) 0.5 0.0))

(test-gpu "predict picks the second class exactly where the decision exceeds the loss's threshold"
  (for* ([dt (in-list dtypes)]
         [loss (in-list losses)])
    (define X (on-matrix blobs dt))
    (law-case (format "~a ~a" dt loss)
      (define r (mbsgd-classifier-fit X (relabel #("B" "M")) #:loss loss))
      (define decisions (device-vector->list (mbsgd-classifier-decision-function r X)))
      (check-equal? (mbsgd-classifier-predict r X)
                    (for/list ([z (in-list decisions)])
                      (if (> z (threshold loss)) "M" "B"))))
    (law-case (format "~a ~a at the threshold" dt loss)
      (define fitted (mbsgd-classifier-fit X (relabel #("B" "M")) #:loss loss))
      (define on-threshold
        (mbsgd-classifier-result (list->device-vector (make-list 3 0.0) #:dtype dt)
                                 (threshold loss)
                                 '("B" "M")
                                 (mbsgd-classifier-result-label-kind fitted)
                                 loss))
      (check-equal? (mbsgd-classifier-predict on-threshold X) (make-list 120 "B")))))

(test-gpu "a result built by hand takes only a fitted label kind, two classes and a known loss"
  (define fitted (mbsgd-classifier-fit (on-matrix blobs) (relabel #(a b))))
  (define coef (mbsgd-classifier-result-coef fitted))
  (define kind (mbsgd-classifier-result-label-kind fitted))
  (define logistic-kind
    (logistic-regression-result-label-kind (logistic-regression-fit (on-matrix blobs) codes)))
  (for ([r (list (row "a symbol label kind"
                      "label-kind: expected the label kind of a fitted result, got 'list"
                      (lambda () (mbsgd-classifier-result coef 0.0 '(a b) 'list 'hinge)))
                 (row "three classes"
                      "classes: expected a list of 2 labels, got '(a b c)"
                      (lambda () (mbsgd-classifier-result coef 0.0 '(a b c) kind 'hinge)))
                 (row "loss 'optimal"
                      "loss: expected one of 'hinge 'log 'squared-loss, got 'optimal"
                      (lambda () (mbsgd-classifier-result coef 0.0 '(a b) kind 'optimal)))
                 (row "a host coef"
                      "coef: expected a device-vector, got '(1.0 2.0 3.0)"
                      (lambda () (mbsgd-classifier-result '(1.0 2.0 3.0) 0.0 '(a b) kind 'hinge)))
                 (row "an exact intercept"
                      "intercept: expected a flonum, got 0"
                      (lambda () (mbsgd-classifier-result coef 0 '(a b) kind 'hinge))))])
    (test-case (row-id r)
      (check-row "mbsgd-classifier-result" r)))
  (test-case "a logistic regression's label kind is a fitted label kind"
    (check-equal? (mbsgd-classifier-predict (mbsgd-classifier-result coef 0.0 '(0 1) logistic-kind 'hinge)
                                            (on-matrix '((0 0 0))))
                  '(0))))

(define big (expt 2 53))

(test-gpu "host labels that unify or collide refuse and class as logistic regression's"
  (define X (on-matrix (for/list ([i (in-range 6)]) (list i (- 6 i) (* i i)))))
  (for ([r (list (row "mixed kinds"
                      "y: expected labels of one kind (reals, strings or symbols), got 0 and \"a\""
                      (lambda () (mbsgd-classifier-fit X '(0 1 0 "a" 1 0))))
                 (row "a non-integral label"
                      "y: expected integral class labels, got 1.5"
                      (lambda () (mbsgd-classifier-fit X '(0 1 0 1.5 1 0))))
                 (row "an infinite label"
                      "y: expected integral class labels, got +inf.0"
                      (lambda () (mbsgd-classifier-fit X '(0 1 0 +inf.0 1 0))))
                 (row "2^53 and 2^53 + 1 given with a flonum collide"
                      "y: expected exactly 2 classes, got 1"
                      (lambda ()
                        (mbsgd-classifier-fit X (list big (add1 big) big 9007199254740992.0 big big)))))])
    (test-case (row-id r)
      (check-row "mbsgd-classifier-fit" r)))
  (test-case "1 and 1.0 unify to flonum classes"
    (check-equal? (mbsgd-classifier-result-classes (mbsgd-classifier-fit X '(0 1 0 1.0 1 0)))
                  '(0.0 1.0)))
  (test-case "exact 2^53 and 2^53 + 1 are two classes"
    (check-equal? (mbsgd-classifier-result-classes
                   (mbsgd-classifier-fit X (list big (add1 big) big (add1 big) big big)))
                  (list big (add1 big)))))
