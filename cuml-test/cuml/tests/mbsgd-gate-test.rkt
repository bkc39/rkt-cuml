#lang racket/base

(require (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in cuml/linear-model mbsgd-classifier-fit mbsgd-regressor-fit)
         (only-in rackunit check-not-exn check-pred)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case on-matrix on-vector)
         (only-in "private/mbsgd-data.rkt"
                  blob-rows
                  clf-parts
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

(define sparse-rows (normal-rows 500 5 3))
(define sparse-targets
  (for/list ([r (in-list sparse-rows)]
             [e (in-list (normal-rows 500 1 4))])
    (+ 1.0 (* 0.3 (car r)) (car e))))
(define-values (overlap overlap-labels) (blob-rows 300 4 5 #:sep 0.3))

(test-gpu "an l1 or elastic-net fit whose optimum is zero passes at a lowered eta0"
  (for* ([dt (in-list dtypes)]
         [penalty (in-list '(l1 elasticnet))])
    (law-case (format "~a ~a regressor" dt penalty)
      (for ([alpha (in-list '(1.0 5.0))])
        (check-pred list?
                    (reg-parts (mbsgd-regressor-fit (on-matrix sparse-rows dt)
                                                    (on-vector sparse-targets dt)
                                                    #:penalty penalty
                                                    #:alpha alpha
                                                    #:l1-ratio 0.9
                                                    #:eta0 1e-5)))))
    (for ([loss (in-list '(hinge log))])
      (law-case (format "~a ~a ~a classifier" dt penalty loss)
        (check-pred list?
                    (clf-parts (mbsgd-classifier-fit (on-matrix overlap dt)
                                                     overlap-labels
                                                     #:loss loss
                                                     #:penalty penalty
                                                     #:alpha 2.0
                                                     #:l1-ratio 0.9
                                                     #:eta0 1e-5)))))))

(define constant-rows (normal-rows 300 4 3))

(test-gpu "a constant target passes: its zero-model objective is round-off"
  (for ([c (in-list '((float64 0.3) (float64 1e6) (float32 1.7)))])
    (match-define (list dt value) c)
    (law-case (format "~a y = ~a" dt value)
      (check-pred list?
                  (reg-parts (mbsgd-regressor-fit (on-matrix constant-rows dt)
                                                  (on-vector (make-list 300 value) dt)))))))
