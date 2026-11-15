#lang racket/base

(require (only-in cuml/linear-model mbsgd-classifier-fit mbsgd-regressor-fit)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-not-exn)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case on-matrix on-vector)
         (only-in "private/logistic-data.rkt" lr-case-X)
         (only-in "private/mbsgd-data.rkt" blob-rows learning-rates normal-rows penalties reg-target)
         (only-in "private/real-logistic-data.rkt" labelled labelled-set)
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
         (mbsgd-classifier-fit Xc labels #:loss loss #:batch-size 1 #:alpha 10 #:eta0 0.01))))))

(define grid-rows (normal-rows 300 5 21))
(define grid-targets (reg-target grid-rows 21))

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

(define sparse-rows (normal-rows 500 5 3))
(define sparse-targets
  (for/list ([r (in-list sparse-rows)]
             [e (in-list (normal-rows 500 1 4))])
    (+ 1.0 (* 0.3 (car r)) (car e))))
(define-values (overlap overlap-labels) (blob-rows 300 4 5 #:sep 0.3))

(test-gpu "an l1 or elastic-net fit whose optimum is zero passes at the default eta0"
  (for* ([dt (in-list dtypes)]
         [penalty (in-list '(l1 elasticnet))])
    (for ([alpha (in-list '(1.0 5.0))])
      (law-case (format "~a ~a regressor alpha ~a" dt penalty alpha)
        (check-not-exn (lambda ()
                         (mbsgd-regressor-fit (on-matrix sparse-rows dt)
                                              (on-vector sparse-targets dt)
                                              #:penalty penalty
                                              #:alpha alpha
                                              #:l1-ratio 0.9)))))
    (for ([loss (in-list '(hinge log))])
      (law-case (format "~a ~a ~a classifier" dt penalty loss)
        (check-not-exn (lambda ()
                         (mbsgd-classifier-fit (on-matrix overlap dt)
                                               overlap-labels
                                               #:loss loss
                                               #:penalty penalty
                                               #:alpha 2.0
                                               #:l1-ratio 0.9)))))))

(test-gpu "an l1 alpha 10 fit on 2 to 10 rows of 3 columns, 20 seeds at 2 and 3, passes at the defaults"
  (for* ([dt (in-list dtypes)]
         [n (in-list '(2 3 4 6 10))])
    (define-values (X y) (blob-rows n 3 (+ 100 n)))
    (for ([loss (in-list '(log hinge squared-loss))])
      (law-case (format "~a ~a classifier on ~a rows" dt loss n)
        (check-not-exn
         (lambda () (mbsgd-classifier-fit (on-matrix X dt) y #:loss loss #:penalty 'l1 #:alpha 10)))))
    (define seeds
      (if (<= n 3)
          (for/list ([s (in-range 20)])
            (+ 1000 (* 37 s) n))
          (list (+ 200 n))))
    (for ([seed (in-list seeds)])
      (define R (normal-rows n 3 seed))
      (law-case (format "~a regressor on ~a rows, seed ~a" dt n seed)
        (check-not-exn (lambda ()
                         (mbsgd-regressor-fit (on-matrix R dt)
                                              (on-vector (reg-target R seed) dt)
                                              #:penalty 'l1
                                              #:alpha 10)))))))

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

(test-gpu "standardised breast cancer, 30 columns, passes an l1 alpha 5 or 10 fit at the defaults"
  (match-define (labelled bc bc-labels _)
    (labelled-set 'breast-cancer "diagnosis" '("id") #:standardise? #t))
  (for* ([dt (in-list dtypes)]
         [loss (in-list '(log hinge))]
         [alpha (in-list '(5 10))])
    (law-case (format "~a ~a alpha ~a" dt loss alpha)
      (check-not-exn (lambda ()
                       (mbsgd-classifier-fit (on-matrix (lr-case-X bc) dt)
                                             bc-labels
                                             #:loss loss
                                             #:penalty 'l1
                                             #:alpha alpha))))))

(define-values (overlap-512 overlap-512-labels) (blob-rows 512 6 45 #:sep 0.3))

(define chance-fits
  `(("hinge l2 alpha 0.1 on overlapping blobs" ,overlap-512 ,overlap-512-labels hinge l2 0.01)
    ("log l1 alpha 0.1 on blobs" ,blobs ,labels log l1 0.1)
    ("hinge l1 alpha 0.1 on blobs" ,blobs ,labels hinge l1 0.05)))

(test-gpu "a one-epoch fit worse than the zero model is refused; eta0/10 passes"
  (for* ([dt (in-list dtypes)]
         [c (in-list chance-fits)])
    (match-define (list name X y loss penalty eta0) c)
    (define (go step)
      (lambda ()
        (mbsgd-classifier-fit (on-matrix X dt)
                              y
                              #:loss loss
                              #:penalty penalty
                              #:alpha 0.1
                              #:eta0 step
                              #:batch-size 1
                              #:epochs 1)))
    (law-case (format "~a ~a" dt name)
      (check-row "mbsgd-classifier-fit" (row name (rose (number->string eta0)) (go eta0)))
      (check-not-exn (go (/ eta0 10))))))

(test-gpu "an l1 rise above five times the zero model is refused however large eta0 makes the allowance"
  (for ([dt (in-list dtypes)])
    (law-case (format "~a log l1 alpha 10 on blobs, eta0 0.3" dt)
      (check-row "mbsgd-classifier-fit"
                 (row "log l1 alpha 10 on blobs, eta0 0.3"
                      (rose "0.3")
                      (lambda ()
                        (mbsgd-classifier-fit (on-matrix blobs dt)
                                              labels
                                              #:loss 'log
                                              #:penalty 'l1
                                              #:alpha 10
                                              #:eta0 0.3
                                              #:learning-rate 'constant
                                              #:batch-size 1
                                              #:epochs 1)))))))

(define constant-rows (normal-rows 300 4 3))

(test-gpu "a constant target passes: its zero-model objective is round-off"
  (for* ([dt (in-list dtypes)]
         [value (in-list '(0.3 1.7 1e6 -1e6 3e7 1e9))])
    (law-case (format "~a y = ~a" dt value)
      (check-not-exn (lambda ()
                       (mbsgd-regressor-fit (on-matrix constant-rows dt)
                                            (on-vector (make-list 300 value) dt)))))))

(define offset-targets
  (for/list ([r (in-list constant-rows)]
             [e (in-list (normal-rows 300 1 9))])
    (+ 1e6 (* 0.05 (car r)) (* 0.05 (car e)))))

(define (offset-fit dt eta0)
  (lambda ()
    (mbsgd-regressor-fit (on-matrix constant-rows dt)
                         (on-vector offset-targets dt)
                         #:batch-size 1
                         #:eta0 eta0)))

(test-gpu "a rise above y's round-off is refused however large y's offset"
  (for ([eta0 (in-list '(0.1 0.2))])
    (law-case (format "float64 eta0 ~a" eta0)
      (check-row
       "mbsgd-regressor-fit"
       (row "y = 1e6 + .05·x + .05·e" (rose (number->string eta0)) (offset-fit 'float64 eta0)))
      (check-not-exn (offset-fit 'float64 (/ eta0 10)))))
  (law-case "float32 eta0 0.2, whose residuals are float32 round-off of 1e6"
    (check-not-exn (offset-fit 'float32 0.2))))
