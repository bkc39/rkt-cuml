#lang racket/base

(require (only-in cuml/linear-model mbsgd-classifier-fit mbsgd-regressor-fit)
         (only-in rackunit check-not-exn check-pred)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case on-matrix on-vector)
         (only-in "private/mbsgd-data.rkt"
                  blob-rows
                  learning-rates
                  normal-rows
                  penalties
                  reg-parts
                  reg-target)
         (only-in "private/refusal-rows.rkt" check-row row row-id))

(define not-finite #rx"^the solution is not finite; lower #:eta0 \\(now 10\\)$")

(define (rose eta0)
  (regexp (string-append "^the objective rose from [0-9.e+-]+ to [0-9.e+-]+; lower #:eta0 \\(now "
                         (regexp-quote eta0)
                         "\\)$")))

(define rows (normal-rows 100 3 41))
(define targets (reg-target rows 41))

(define (diverging dt lr)
  (lambda ()
    (mbsgd-regressor-fit (on-matrix rows dt) (on-vector targets dt) #:eta0 10 #:learning-rate lr)))

(test-gpu "a diverging fit is refused: a float32 NaN as not finite, a huge float64 as a rise"
  (for ([r (list (row "float32 constant" not-finite (diverging 'float32 'constant))
                 (row "float64 constant" (rose "10") (diverging 'float64 'constant))
                 (row "float32 invscaling" not-finite (diverging 'float32 'invscaling))
                 (row "float64 invscaling" (rose "10") (diverging 'float64 'invscaling))
                 (row "float32 adaptive" not-finite (diverging 'float32 'adaptive)))])
    (law-case (row-id r)
      (check-row "mbsgd-regressor-fit" r))))

(define near-zero (normal-rows 512 6 42))
(define near-zero-targets (reg-target near-zero 42))
(define-values (blobs labels) (blob-rows 512 6 43))

(test-gpu "a fit ending above the zero model is refused, and the advised lower eta0 passes"
  (define X (on-matrix near-zero))
  (define y (on-vector near-zero-targets))
  (law-case "regressor"
    (check-row "mbsgd-regressor-fit"
               (row "eta0 0.05"
                    (rose "0.05")
                    (lambda () (mbsgd-regressor-fit X y #:batch-size 1 #:alpha 10 #:eta0 0.05))))
    (check-not-exn (lambda () (mbsgd-regressor-fit X y #:batch-size 1 #:alpha 10 #:eta0 0.005))))
  (for ([loss (in-list '(log hinge))])
    (law-case (format "~a classifier" loss)
      (define Xc (on-matrix blobs))
      (check-row
       "mbsgd-classifier-fit"
       (row "eta0 0.05"
            (rose "0.05")
            (lambda ()
              (mbsgd-classifier-fit Xc labels #:loss loss #:batch-size 1 #:alpha 10 #:eta0 0.05))))
      (check-not-exn
       (lambda ()
         (mbsgd-classifier-fit Xc labels #:loss loss #:batch-size 1 #:alpha 10 #:eta0 0.005))))))

(define grid-rows (normal-rows 300 5 21))
(define grid-targets (reg-target grid-rows 21))

(test-gpu "no documented case is refused by the gate"
  (for* ([dt (in-list dtypes)]
         [intercept? (in-list '(#t #f))]
         [lr (in-list learning-rates)]
         [penalty (in-list penalties)])
    (law-case (format "~a fit-intercept ~a ~a ~a" dt intercept? lr penalty)
      (check-pred list?
                  (reg-parts (mbsgd-regressor-fit (on-matrix grid-rows dt)
                                                  (on-vector grid-targets dt)
                                                  #:fit-intercept intercept?
                                                  #:learning-rate lr
                                                  #:penalty penalty
                                                  #:batch-size 16
                                                  #:eta0 0.01
                                                  #:epochs 50))))))
