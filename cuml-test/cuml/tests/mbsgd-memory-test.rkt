#lang racket/base

(require (only-in cuml/linear-model
                  mbsgd-classifier-fit
                  mbsgd-classifier-predict
                  mbsgd-regressor-fit
                  mbsgd-regressor-predict)
         (only-in "private/linear-data.rkt" on-matrix on-vector)
         (only-in "private/mbsgd-data.rkt" blob-rows normal-rows reg-target)
         (only-in "private/process-memory.rkt" test-fit-memory))

(define (setup)
  (define rows (normal-rows 2000 7 61))
  (define-values (blobs labels) (blob-rows 2000 7 62))
  (define X (on-matrix rows 'float32))
  (define y (on-vector (reg-target rows 61) 'float32))
  (define B (on-matrix blobs 'float32))
  (define codes (on-vector labels 'int32))
  (define small (normal-rows 100 3 41))
  (define diverging-X (on-matrix small 'float32))
  (define diverging-y (on-vector (reg-target small 41) 'float32))
  (define small-X (on-matrix small))
  (define huge-y
    (on-vector (for/list ([i (in-range 100)])
                 (if (even? i) 1e155 -1e155))))
  (define huge-float32-y
    (on-vector (for/list ([i (in-range 100)])
                 (if (even? i) 3e38 -3e38))
               'float32))
  (define three-classes
    (on-vector (for/list ([i (in-range 100)])
                 (modulo i 3))
               'int32))
  (values (lambda ()
            (mbsgd-regressor-predict (mbsgd-regressor-fit X y #:epochs 20) X)
            (mbsgd-classifier-predict (mbsgd-classifier-fit B codes #:epochs 20) B))
          (list (list "mbsgd-regressor-fit: the solution is not finite; lower #:eta0 (now 10)"
                      (lambda () (mbsgd-regressor-fit diverging-X diverging-y #:eta0 10)))
                (list "mbsgd-regressor-fit: y is too large to check the fit; rescale y"
                      (lambda () (mbsgd-regressor-fit small-X huge-y)))
                (list "mbsgd-regressor-fit: y is too large to check the fit; rescale y"
                      (lambda () (mbsgd-regressor-fit diverging-X huge-float32-y #:eta0 1e-30)))
                (list "mbsgd-classifier-fit: y: expected exactly 2 classes, got 3"
                      (lambda () (mbsgd-classifier-fit diverging-X three-classes))))))

(define fit-and-predict-buffers 9)

(test-fit-memory "mbsgd"
                 #:rounds 50
                 #:sample-every 10
                 #:setup setup
                 #:buffers-per-round fit-and-predict-buffers)
