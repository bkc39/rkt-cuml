#lang racket/base

(require (only-in cuml/linear-model mbsgd-regressor-fit)
         (only-in rackunit check-not-exn)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/common-data.rkt" on-vector)
         (only-in "private/linear-data.rkt" dtypes law-case on-matrix)
         (only-in "private/mbsgd-data.rkt" learning-rates penalties reg-target)
         (only-in "private/mbsgd-gate-data.rkt" grid-rows grid-targets))

(test-gpu "no documented case is refused by the gate"
  (for* ([dt (in-list dtypes)]
         [intercept? (in-list '(#t #f))]
         [lr (in-list learning-rates)]
         [penalty (in-list penalties)])
    (law-case (format "~a fit-intercept ~a ~a ~a" dt intercept? lr penalty)
      (check-not-exn (lambda ()
                       (mbsgd-regressor-fit (on-matrix grid-rows dt)
                                            (on-vector grid-targets dt)
                                            #:fit-intercept intercept?
                                            #:learning-rate lr
                                            #:penalty penalty
                                            #:batch-size 16
                                            #:eta0 0.01
                                            #:epochs 50))))))

(define small-targets
  (for/list ([t (in-list (reg-target grid-rows 21 #:offset 0.0))])
    (* 0.01 t)))

(test-gpu "a regressor l1 alpha 1 fit on y scaled by 0.01 passes at eta0 1e-3"
  (for ([dt (in-list dtypes)])
    (law-case (symbol->string dt)
      (check-not-exn (lambda ()
                       (mbsgd-regressor-fit (on-matrix grid-rows dt)
                                            (on-vector small-targets dt)
                                            #:penalty 'l1
                                            #:alpha 1
                                            #:eta0 1e-3))))))
