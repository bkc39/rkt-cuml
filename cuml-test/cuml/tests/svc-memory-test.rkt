#lang racket/base

(require (only-in cuml/svm svc-decision-function svc-fit svc-predict)
         (only-in racket/list make-list)
         (only-in "private/common-data.rkt" on-vector)
         (only-in "private/linear-data.rkt" on-matrix)
         (only-in "private/process-memory.rkt" test-fit-memory)
         (only-in "private/svm-data.rkt" dyadic-blobs))

(define (setup)
  (define-values (rows labels) (dyadic-blobs 2000 6 41))
  (define X (on-matrix rows 'float32))
  (define w
    (on-vector (for/list ([i (in-range 2000)])
                 (if (even? i) 1.0 2.0))
               'float32))
  (define-values (small-rows small-labels) (dyadic-blobs 40 3 17))
  (define small (on-matrix small-rows 'float32))
  (define one-weight (on-vector (cons -1.0 (make-list 39 1.0)) 'float32))
  (values (lambda ()
            (define r (svc-fit X labels #:kernel 'rbf #:sample-weight w))
            (svc-decision-function r X)
            (svc-predict r X))
          (list (list "svc-fit: sample-weight: negative weight -1 at row 0"
                      (lambda () (svc-fit small small-labels #:sample-weight one-weight)))
                (list #rx"^svc-fit: the kernel can overflow "
                      (lambda () (svc-fit small small-labels #:gamma 1e30)))
                (list "svc-fit: the fit has no support vectors; lower #:tol or raise #:max-iter"
                      (lambda () (svc-fit small small-labels #:tol 3))))))

(define fit-and-apply-buffers 7)

(define encoded-codes 1)

(test-fit-memory "svc"
                 #:rounds 100
                 #:sample-every 10
                 #:setup setup
                 #:buffers-per-round fit-and-apply-buffers
                 #:buffers-per-refusal encoded-codes)
