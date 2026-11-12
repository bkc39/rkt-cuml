#lang racket/base

(require (only-in cuml elastic-net fit lasso)
         (only-in cuml/linear-model
                  elastic-net-fit
                  elastic-net-params
                  elastic-net-predict
                  lasso-fit
                  lasso-predict
                  lasso-result
                  linear-regression-result)
         (only-in racket/list filter-not)
         (only-in racket/match match-define match-lambda match-lambda** struct*)
         (only-in rackunit check-equal? check-exn check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" normal-source)
         (only-in "private/cuml-error.rkt" check-cuml-error raised)
         (only-in "private/elastic-net-data.rkt"
                  apply-keywords
                  case-alpha
                  check-accuracy
                  coef-distance
                  converged?
                  correlated-case
                  default-tol
                  en-case-fit
                  en-run
                  en-run-coef
                  en-run-n-iter
                  en-run?
                  keyword-hash)
         (only-in "private/elastic-net-oracle.rkt"
                  en-solution
                  host-kkt
                  kkt
                  oracle-elastic-net
                  problem-of)
         (only-in "private/kmeans-data.rkt" blank-matrix)
         (only-in "private/linear-data.rkt"
                  docs-case
                  dtypes
                  dyadic
                  lcase
                  lcase-w
                  lcase-X
                  lcase-y
                  max-abs
                  [on-matrix matrix]
                  [on-vector vector]
                  scaled-case
                  scaled-columns
                  unit-roundoff)
         (only-in "private/linear-refusal-rows.rkt" grid shared-fit-rows targets)
         (only-in "private/refusal-rows.rkt" check-row row row-id))

(define max-iter-message "max-iter: expected an exact integer from 1 to 2147483647, got ")

(define (l1-ratio-rows go X y bad-X)
  (list (row "non-real l1-ratio before max-iter"
             "l1-ratio: expected a real number, got 'x"
             (lambda () (go bad-X y #:l1-ratio 'x #:max-iter 0)))
        (row "l1-ratio below 0"
             "l1-ratio: expected a finite value from 0 to 1, got -0.5"
             (lambda () (go X y #:l1-ratio -0.5)))
        (row "l1-ratio above 1"
             "l1-ratio: expected a finite value from 0 to 1, got 1.5"
             (lambda () (go X y #:l1-ratio 1.5)))
        (row "nan l1-ratio"
             "l1-ratio: expected a finite value from 0 to 1, got nan"
             (lambda () (go X y #:l1-ratio +nan.0)))))

(define (fit-rows go #:l1-ratio? [l1-ratio? #t])
  (define X (matrix grid))
  (define y (vector targets))
  (define bad-X (blank-matrix 5 3 'int32))
  (append (list (row "non-boolean fit-intercept before every other fault"
                     "fit-intercept: expected #t or #f, got 'yes"
                     (lambda () (go bad-X y #:fit-intercept 'yes #:alpha 'x #:tol 'x)))
                (row "non-real alpha before l1-ratio"
                     "alpha: expected a real number, got 'x"
                     (lambda () (go bad-X y #:alpha 'x #:max-iter 0 #:tol 'x)))
                (row "max-iter 0 before tol"
                     (string-append max-iter-message "0")
                     (lambda () (go bad-X y #:max-iter 0 #:tol 'x)))
                (row "max-iter past int32"
                     (string-append max-iter-message "2147483648")
                     (lambda () (go X y #:max-iter (expt 2 31))))
                (row "non-integer max-iter"
                     (string-append max-iter-message "1.5")
                     (lambda () (go X y #:max-iter 1.5)))
                (row "negative max-iter"
                     (string-append max-iter-message "-1")
                     (lambda () (go X y #:max-iter -1)))
                (row "non-real tol before the X views"
                     "tol: expected a real number, got 'x"
                     (lambda () (go bad-X y #:tol 'x)))
                (row "X dtype before selection"
                     "X: expected float32 or float64, got int32"
                     (lambda () (go bad-X y #:selection 'shuffled)))
                (row "unknown selection before alpha"
                     "selection: expected cyclic or random"
                     (lambda () (go X y #:selection 'shuffled #:alpha -1)))
                (row "negative alpha before l1-ratio"
                     "alpha: expected a finite value at least 0, got -1"
                     (lambda () (go X y #:alpha -1 #:tol -1)))
                (row "infinite alpha"
                     "alpha: expected a finite value at least 0, got inf"
                     (lambda () (go X y #:alpha +inf.0)))
                (row "nan alpha"
                     "alpha: expected a finite value at least 0, got nan"
                     (lambda () (go X y #:alpha +nan.0)))
                (row "negative tol"
                     "tol: expected a finite value of at least 0, got -1"
                     (lambda () (go X y #:tol -1)))
                (row "nan tol"
                     "tol: expected a finite value of at least 0, got nan"
                     (lambda () (go X y #:tol +nan.0)))
                (row "infinite tol"
                     "tol: expected a finite value of at least 0, got inf"
                     (lambda () (go X y #:tol +inf.0)))
                (row "tol before a non-finite X"
                     "tol: expected a finite value of at least 0, got -1"
                     (lambda () (go (matrix '((1 2) (+nan.0 1) (2 2))) (vector '(1 2 3)) #:tol -1))))
          (if l1-ratio?
              (l1-ratio-rows go X y bad-X)
              '())
          (filter-not (lambda (r) (equal? (row-id r) "intercept overflow")) (shared-fit-rows go))))

(define (estimator-fit make)
  (make-keyword-procedure (lambda (kws vals X y)
                            (define given (keyword-hash kws vals))
                            (fit (apply-keywords make (hash-remove given '#:sample-weight) '())
                                 X
                                 y
                                 #:sample-weight (hash-ref given '#:sample-weight #f)))))

(test-gpu "elastic-net-fit refuses each row, the earlier fault first"
  (for ([r (in-list (fit-rows elastic-net-fit))])
    (test-case (row-id r)
      (check-row "elastic-net-fit" r))))

(test-gpu "lasso-fit refuses each row but l1-ratio's"
  (for ([r (in-list (fit-rows lasso-fit #:l1-ratio? #f))])
    (test-case (row-id r)
      (check-row "lasso-fit" r))))

(test-gpu "the estimators' fit names the generic"
  (for ([r (in-list (fit-rows (estimator-fit elastic-net)))])
    (test-case (row-id r)
      (check-row "fit" r))))

(test-gpu "predict refuses a result of another kind, the estimators a missing y"
  (define X (matrix grid))
  (check-exn #rx"^elastic-net-predict: contract violation\n  expected: elastic-net-result\\?"
             (lambda () (elastic-net-predict (lasso-result (vector '(1 2 3)) 0.0 1) X)))
  (check-exn #rx"^lasso-predict: contract violation\n  expected: lasso-result\\?"
             (lambda () (lasso-predict (linear-regression-result (vector '(1 2 3)) 0.0) X)))
  (check-cuml-error 'logic
                    "lasso-predict: X: expected 3 columns, got 2"
                    (lambda () (lasso-predict (lasso-fit X (vector targets)) (matrix '((1 2))))))
  (check-cuml-error 'logic "fit: lasso needs y" (lambda () (fit (lasso) X)))
  (check-cuml-error 'logic "fit: elastic-net needs y" (lambda () (fit (elastic-net) X)))
  (check-cuml-error 'logic
                    "elastic-net-params: elastic-net has no parameter solver"
                    (lambda () (elastic-net-params (hasheq 'solver 'cd)))))

(test-gpu "an absolute stop short of the relative test is refused; y scaled up fits"
  (define wide (scaled-case docs-case #:X 100000))
  (define c (scaled-case (correlated-case 500 8 0.9 21) #:X 100000))
  (define alpha (case-alpha c 0.01 1))
  (for ([dt (in-list dtypes)])
    (check-cuml-error
     'logic
     #rx"^elastic-net-fit: X: coordinate descent stopped after 1 epochs with every coefficient under tol 0.001 and KKT residual "
     (lambda () (en-case-fit wide dt #:alpha 1e4 #:l1-ratio 1)))
    (check-cuml-error 'logic
                      #rx"; tol is absolute in the coefficients' units, so scale y up or X down$"
                      (lambda () (en-case-fit c dt #:alpha alpha #:l1-ratio 1)))
    (define up (scaled-case c #:y (expt 2 17)))
    (define alpha-up (case-alpha up 0.01 1))
    (check-true (converged? (en-case-fit up dt #:alpha alpha-up #:l1-ratio 1)))
    (check-accuracy up dt alpha-up 1 1e-3))
  (check-cuml-error 'logic
                    #rx"^elastic-net-fit: X: coordinate descent stopped after "
                    (lambda ()
                      (en-case-fit (scaled-case docs-case #:X (expt 10 20)) 'float32 #:alpha 1e-3))))

(test-gpu "nan, infinite and out-of-range hyperparameters are refused"
  (define c (correlated-case 20 2 0 22))
  (for ([alpha (in-list '(+nan.0 +inf.0 -1.0))])
    (check-cuml-error 'logic
                      #rx"^elastic-net-fit: alpha: expected a finite value at least 0, got "
                      (lambda () (en-case-fit c 'float64 #:alpha alpha))))
  (for ([r (in-list '(-0.5 1.5 +nan.0))])
    (check-cuml-error 'logic
                      #rx"^elastic-net-fit: l1-ratio: expected a finite value from 0 to 1, got "
                      (lambda () (en-case-fit c 'float64 #:alpha 0.1 #:l1-ratio r))))
  (for ([tol (in-list '(-1 +nan.0 +inf.0))])
    (check-cuml-error 'logic
                      #rx"^elastic-net-fit: tol: expected a finite value of at least 0, got "
                      (lambda () (en-case-fit c 'float64 #:alpha 0.1 #:tol tol))))
  (for ([k (in-list (list 0 -1 (expt 2 31) 1.5))])
    (check-cuml-error 'logic
                      (regexp (string-append "^elastic-net-fit: " max-iter-message))
                      (lambda () (en-case-fit c 'float64 #:alpha 0.1 #:max-iter k))))
  (check-equal? (en-run-n-iter (en-case-fit c 'float64 #:alpha 0.1 #:tol 0 #:max-iter 9)) 9)
  (check-true (> (en-run-n-iter (en-case-fit c 'float64 #:alpha 0)) 1)))

(test-gpu "float32 columns 2^120 apart are too narrow; 2^60 apart fit"
  (define c (correlated-case 100 2 0 23))
  (check-cuml-error
   'logic
   "elastic-net-fit: X: column 0 is too narrow beside the other columns and y for float32 coordinate descent"
   (lambda ()
     (en-case-fit (scaled-columns c (list (expt 2 -60) (expt 2 60)) 1) 'float32 #:alpha 1e-3)))
  (check-accuracy (scaled-columns c (list (expt 2 -30) (expt 2 30)) 1) 'float32 1e-3 0.5 1e-3))

(test-gpu "an alpha whose row multiple overflows float32 is refused as ridge's"
  (check-cuml-error
   'logic
   #rx"^elastic-net-fit: alpha: expected at most .* for this X in float32, got 1e\\+38$"
   (lambda () (en-case-fit (correlated-case 20 2 0 24) 'float32 #:alpha 1e38))))

(define (columns-apart narrow)
  (define c (correlated-case 100 2 0 23))
  (lcase "a narrow column beside one 2^28 wide"
         (map (match-lambda
                [(list x0 x1) (list (* (expt 2 narrow) x0) (* (expt 2 28) x1))])
              (lcase-X c))
         (map (match-lambda**
                [((list _ x1) v) (+ x1 (/ v 1000))])
              (lcase-X c)
              (lcase-y c))
         #f))

(test-gpu "an alpha past half the row-multiple range is refused"
  (for ([dt (in-list dtypes)]
        [narrow (in-list '(-30 -480))]
        [limit (in-list '(94447329657392906240 23611832414348222464))]
        [past (in-list '(1.888757e20 4.721894e19))])
    (define c (columns-apart narrow))
    (check-cuml-error
     'logic
     (regexp (format "^elastic-net-fit: alpha: expected at most ~a for this X in ~a, got " limit dt))
     (lambda () (en-case-fit c dt #:alpha past #:l1-ratio 1e-13)))
    (define (scaled-wide-coef alpha)
      (match-define (en-run (list _ b1) _ _) (en-case-fit c dt #:alpha alpha #:l1-ratio 1e-13))
      (* alpha b1))
    (define at-limit (scaled-wide-coef (* limit (- 1 1e-4))))
    (check-true (> at-limit 0))
    (check-true (< (abs (- at-limit (scaled-wide-coef (* limit 0.99)))) (* 1e-3 at-limit)))))

(test-gpu "the alpha bound a refusal prints fits"
  (for ([dt (in-list dtypes)]
        [narrow (in-list '(-30 -480))])
    (define c (columns-apart narrow))
    (define message
      (exn-message (raised (lambda () (en-case-fit c dt #:alpha 1e300 #:l1-ratio 1e-13)))))
    (define printed (string->number (cadr (regexp-match #rx"at most ([^ ]+) for" message))))
    (check-true (en-run? (en-case-fit c dt #:alpha printed #:l1-ratio 1e-13)))))

(define stop-message
  #rx"^elastic-net-fit: X: coordinate descent stopped after 1 epochs with every coefficient under tol ")

(test-gpu "under a heavy L2 penalty, an absolute stop short of tol is refused"
  (define c (scaled-case (correlated-case 200 7 0.9 31) #:y (expt 2 -10)))
  (for ([dt (in-list dtypes)])
    (check-cuml-error 'logic
                      stop-message
                      (lambda () (en-case-fit c dt #:alpha 1e4 #:l1-ratio 0 #:tol 1e-5)))))

(define tall-rows (expt 2 17))

(define nearly-orthogonal
  (let ([X (for/list ([i (in-range tall-rows)])
             (define a (if (even? i) 1 -1))
             (define b (if (even? (quotient i 2)) 1 -1))
             (list a (* (expt 2 10) (+ b (* (expt 2 -8) a)))))])
    (lcase "a column 2^-8 from orthogonal to one 2^10 wider"
           X
           (map (match-lambda
                  [(list x0 x1) (+ (* (expt 2 -21) x0) (* (expt 2 -30) x1))])
                X)
           #f)))

(test-gpu "a float32 one-epoch stop 0.8% off is refused at 2^17 rows"
  (check-cuml-error 'logic
                    stop-message
                    (lambda () (en-case-fit nearly-orthogonal 'float32 #:alpha 0 #:tol 1e-6)))
  (match-define (en-run (list b0 _) _ 1)
    (en-case-fit nearly-orthogonal 'float32 #:alpha 0 #:tol 1e-6 #:max-iter 1))
  (check-true (> (abs (- (inexact->exact b0) (expt 2 -21))) (* 1e-3 (expt 2 -21)))))

(define weak-signal
  (let* ([next (normal-source 41868)]
         [X (for/list ([_ (in-range (expt 2 15))])
              (define shared (next))
              (for/list ([_ (in-range 4)])
                (dyadic (+ (* (sqrt 0.9) shared) (* (sqrt 0.1) (next))))))])
    (lcase "four columns at correlation 0.9 and a y of pure noise"
           X
           (for/list ([_ (in-list X)])
             (dyadic (next)))
           #f)))

(test-gpu "a float32 absolute stop over tol within its rounding is refused; a lower tol fits"
  (check-cuml-error
   'logic
   (pregexp (string-append "^elastic-net-fit: X: coordinate descent stopped after 1 epochs with "
                           "every coefficient under tol 0\\.001 and KKT residual [^ ]+ > tol, "
                           "within float32's rounding of the solve; lower tol below the largest "
                           "coefficient \\([^)]+\\) or fit in float64$"))
   (lambda () (en-case-fit weak-signal 'float32 #:alpha 1500 #:l1-ratio 0)))
  (define (fit-at tol #:max-iter [max-iter 1000])
    (en-case-fit weak-signal 'float32 #:alpha 1500 #:l1-ratio 0 #:tol tol #:max-iter max-iter))
  (define fitted (fit-at 1e-7))
  (check-true (converged? fitted))
  (check-true (> (coef-distance (en-run-coef (fit-at 1e-3 #:max-iter 1)) (en-run-coef fitted))
                 (* 1e-3 (max-abs (en-run-coef fitted))))))

(define unit-scales '(1 30 1000))

(define mixed-units
  (let* ([next (normal-source 7)]
         [X (for/list ([_ (in-range tall-rows)])
              (define shared (next))
              (for/list ([scale (in-list unit-scales)])
                (dyadic (* scale (+ (* (sqrt 0.6) shared) (* (sqrt 0.4) (next)))))))])
    (lcase "three columns at correlation 0.6 in units 1, 30 and 1000, and a weak signal"
           X
           (for/list ([row (in-list X)])
             (dyadic (+ (for/sum ([x (in-list row)] [scale (in-list unit-scales)])
                                 (* 1/1000 (/ x scale)))
                        (next))))
           #f)))

(define (check-accepted-within-tol c dt alpha r)
  (define p (problem-of (lcase-X c) (lcase-y c) (lcase-w c) #t))
  (match-define (en-run coef _ n-iter) (en-case-fit c dt #:alpha alpha #:l1-ratio r))
  (check-equal? n-iter 1)
  (match-define (kkt K B A)
    (host-kkt p coef alpha r default-tol n-iter (length (lcase-y c)) (unit-roundoff dt)))
  (check-true (<= K B))
  (check-true (> A B))
  (match-define (struct* en-solution ([coef c*])) (oracle-elastic-net p alpha r))
  (check-true (< (coef-distance coef c*) (* default-tol (max-abs c*)))))

(test-gpu "a float32 absolute stop within tol is accepted where its rounding exceeds tol"
  (check-accepted-within-tol mixed-units 'float32 (case-alpha mixed-units 0.1 0.2) 0.2))

(test-gpu "a float64 lasso stop 1e-12 below alpha-max is accepted within tol where its A exceeds tol"
  (define c (correlated-case (expt 2 16) 4 0.5 11))
  (check-accepted-within-tol c 'float64 (case-alpha c (- 1 1e-12) 1) 1))
