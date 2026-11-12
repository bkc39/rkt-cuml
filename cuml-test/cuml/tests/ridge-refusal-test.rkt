#lang racket/base

(require (only-in cuml fit ridge)
         (only-in cuml/linear-model linear-regression-result ridge-fit ridge-params ridge-predict)
         (only-in racket/match match-define struct*)
         (only-in rackunit check-equal? check-exn check-true test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/kmeans-data.rkt" blank-matrix)
         (only-in "private/linear-data.rkt"
                  check-law
                  dtypes
                  exact-fit
                  gap
                  lcase
                  lcase-name
                  lcase-X
                  lcase-y
                  max-abs
                  normal-case
                  oracle-ridge
                  [on-matrix matrix]
                  [on-vector vector]
                  ridge-solution
                  rounded
                  unit-roundoff
                  weighted-spread)
         (only-in "private/linear-refusal-rows.rkt"
                  grid
                  overflow-X
                  overflow-y
                  shared-fit-rows
                  targets)
         (only-in "private/refusal-rows.rkt" check-row row row-id)
         (only-in "private/ridge-data.rkt" ridge-case-fit solvers))

(define (fit-rows go)
  (define X (matrix grid))
  (define y (vector targets))
  (append (list (row "non-boolean fit-intercept before the X views"
                     "fit-intercept: expected #t or #f, got 'yes"
                     (lambda () (go (blank-matrix 5 3 'int32) y #:fit-intercept 'yes #:alpha 'x)))
                (row "non-real alpha before the X views"
                     "alpha: expected a real number, got 'x"
                     (lambda () (go (blank-matrix 5 3 'int32) y #:alpha 'x)))
                (row "X dtype before alpha"
                     "X: expected float32 or float64, got int32"
                     (lambda () (go (blank-matrix 5 3 'int32) y #:alpha -1)))
                (row "unknown solver"
                     "solver: expected auto, eig or svd"
                     (lambda () (go X y #:solver 'lsmr)))
                (row "unknown solver before alpha"
                     "solver: expected auto, eig or svd"
                     (lambda () (go X y #:solver 'lsmr #:alpha -1)))
                (row "negative alpha"
                     "alpha: expected a finite value at least 0, got -1"
                     (lambda () (go X y #:alpha -1)))
                (row "infinite alpha"
                     "alpha: expected a finite value at least 0, got inf"
                     (lambda () (go X y #:alpha +inf.0)))
                (row "nan alpha"
                     "alpha: expected a finite value at least 0, got nan"
                     (lambda () (go X y #:alpha +nan.0)))
                (row "alpha past the double range"
                     "alpha: expected a finite value at least 0, got inf"
                     (lambda () (go X y #:alpha (expt 10 400))))
                (row "alpha before a non-finite y"
                     "alpha: expected a finite value at least 0, got -1"
                     (lambda () (go X (vector '(1 2 3 4 +inf.0)) #:alpha -1)))
                (row "coefficient overflow"
                     "X: column 2: the coefficient exceeds the float32 range"
                     (lambda () (go (overflow-X) (overflow-y) #:alpha (expt 2 -250)))))
          (shared-fit-rows go)))

(define (estimator-fit X
                       y
                       #:sample-weight [w #f]
                       #:alpha [alpha 1.0]
                       #:solver [s 'auto]
                       #:fit-intercept [fi #t])
  (fit (ridge #:alpha alpha #:solver s #:fit-intercept fi) X y #:sample-weight w))

(test-gpu "ridge-fit refuses each row, the earlier fault first (Q1)"
  (for ([r (in-list (fit-rows ridge-fit))])
    (test-case (row-id r)
      (check-row "ridge-fit" r))))

(test-gpu "the estimator's fit names the generic"
  (for ([r (in-list (fit-rows estimator-fit))])
    (test-case (row-id r)
      (check-row "fit" r))))

(test-gpu "ridge-predict refuses a result of another kind, and the estimator a missing y"
  (check-exn #rx"^ridge-predict: contract violation\n  expected: ridge-result\\?"
             (lambda ()
               (ridge-predict (linear-regression-result (vector '(1 2 3)) 0.0) (matrix grid))))
  (check-cuml-error 'logic
                    "ridge-predict: X: expected 3 columns, got 2"
                    (lambda ()
                      (ridge-predict (ridge-fit (matrix grid) (vector targets)) (matrix '((1 2))))))
  (check-cuml-error 'logic "fit: ridge needs y" (lambda () (fit (ridge) (matrix grid))))
  (check-cuml-error 'logic
                    "ridge-params: ridge has no parameter algorithm"
                    (lambda () (ridge-params (hasheq 'algorithm 'eig)))))

(test-gpu "alpha 0, 0.0 and -0.0 are accepted; inf, nan and -1 are not (Q3)"
  (for ([alpha (in-list '(0 0.0 -0.0))])
    (check-true (list? (ridge-case-fit (normal-case 20 2 0) 'float64 #:alpha alpha))))
  (for ([alpha (in-list '(+inf.0 +nan.0 -1.0))])
    (check-cuml-error 'logic
                      #rx"^ridge-fit: alpha: expected a finite value at least 0, got "
                      (lambda () (ridge-case-fit (normal-case 20 2 0) 'float64 #:alpha alpha)))))

(define mixed-scales
  (let ([c (normal-case 60 4 11)])
    (lcase "columns at 2^-13, 1, 2^13, 1"
           (for/list ([row (in-list (lcase-X c))])
             (map * row (list (expt 2 -13) 1 (expt 2 13) 1)))
           (lcase-y c)
           #f)))

(test-gpu "on mixed scales float32 eig is refused and svd fits (Q2)"
  (check-cuml-error 'logic
                    #rx"^ridge-fit: X: the eig solve is not a ridge solution "
                    (lambda () (ridge-case-fit mixed-scales 'float32 #:alpha 1 #:solver 'eig)))
  (match-define (struct* ridge-solution ([coef c*]))
    (oracle-ridge (lcase-X mixed-scales) (lcase-y mixed-scales) #f 1 #t))
  (match-define (list coef _)
    (exact-fit (ridge-case-fit mixed-scales 'float32 #:alpha 1 #:solver 'svd)))
  (check-law "svd gap" (gap coef c*) (* (expt 2 -10) (max-abs c*))))

(define (scales-apart dt g)
  (define c (normal-case 200 3 5))
  (define (in-dtype x)
    (inexact->exact (rounded dt x)))
  (lcase (format "columns at 2^~a and 2^-~a" g g)
         (for/list ([row (in-list (lcase-X c))])
           (match-define (list a b _) row)
           (list (in-dtype (* (expt 2 g) a)) (in-dtype (* (expt 2 (- g)) b))))
         (for/list ([row (in-list (lcase-X c))]
                    [yi (in-list (lcase-y c))])
           (match-define (list a b _) row)
           (in-dtype (+ a b (* 1/10 (- yi 5)))))
         #f))

(define (spread-alpha column k)
  (~> (* (expt 2 k) (weighted-spread column (map (lambda (_) 1) column)))
      real->double-flonum
      inexact->exact))

(define (narrow-alpha c k)
  (spread-alpha (map cadr (lcase-X c)) k))

(test-gpu "a narrow column's coefficient lost under a large alpha is refused"
  (for* ([p (in-list '((float32 44 30) (float64 100 40) (float32 60 0)))]
         [s (in-list solvers)]
         [fi (in-list '(#t #f))])
    (match-define (list dt g k) p)
    (define c (scales-apart dt g))
    (test-case (format "~a ~a alpha 2^~a ~a ~a" dt (lcase-name c) k s fi)
      (check-cuml-error
       'logic
       #rx"^ridge-fit: X: the (eig|svd) solve is not a ridge solution .*; X is ill-conditioned for this alpha$"
       (lambda () (ridge-case-fit c dt #:alpha (narrow-alpha c k) #:solver s #:fit-intercept fi))))))

(test-gpu "columns 2^40 apart in spread fit under a large alpha"
  (for* ([s (in-list solvers)]
         [fi (in-list '(#t #f))])
    (define c (scales-apart 'float32 20))
    (define alpha (narrow-alpha c 20))
    (test-case (format "~a ~a" s fi)
      (match-define (struct* ridge-solution ([coef c*]))
        (oracle-ridge (lcase-X c) (lcase-y c) #f alpha fi))
      (match-define (list coef _)
        (exact-fit (ridge-case-fit c 'float32 #:alpha alpha #:solver s #:fit-intercept fi)))
      (check-law "gap" (gap coef c*) (* 64 (unit-roundoff 'float32) (max-abs c*))))))

(test-gpu "X, y and w are unchanged after every fit"
  (for* ([dt (in-list dtypes)]
         [s (in-list solvers)]
         [w (in-list (list #f '(1 0 2 3 1/2)))]
         [fi (in-list '(#t #f))])
    (define X (matrix grid dt))
    (define y (vector targets dt))
    (define W (and w (vector w dt)))
    (define (snapshot)
      (list (device-matrix->list* X) (device-vector->list y) (and W (device-vector->list W))))
    (define before (snapshot))
    (ridge-fit X y #:sample-weight W #:alpha 1/2 #:solver s #:fit-intercept fi)
    (test-case (format "~a ~a ~a ~a" dt s (and w #t) fi)
      (check-equal? (snapshot) before))))

(define (one-wide-two-narrow g)
  (define base (normal-case 200 3 5))
  (define (in-dtype x)
    (inexact->exact (rounded 'float32 x)))
  (lcase (format "a at 2^~a, c + a/2 and c + b at 2^-~a" g g)
         (for/list ([row (in-list (lcase-X base))])
           (match-define (list a b c) row)
           (map in-dtype
                (list (* (expt 2 g) a) (* (expt 2 (- g)) (+ c (/ a 2))) (* (expt 2 (- g)) (+ c b)))))
         (for/list ([row (in-list (lcase-X base))]
                    [yi (in-list (lcase-y base))])
           (match-define (list a b c) row)
           (in-dtype (+ a (+ c (/ a 2)) (+ c b) (* 1/10 (- yi 5)))))
         #f))

(test-gpu "three correlated columns 2^40 apart in spread fit svd under a large alpha"
  (define c (one-wide-two-narrow 10))
  (define alpha (spread-alpha (map car (lcase-X c)) 10))
  (for ([fi (in-list '(#t #f))])
    (test-case (format "~a ~a" (lcase-name c) fi)
      (match-define (struct* ridge-solution ([coef c*]))
        (oracle-ridge (lcase-X c) (lcase-y c) #f alpha fi))
      (match-define (list coef _)
        (exact-fit (ridge-case-fit c 'float32 #:alpha alpha #:solver 'svd #:fit-intercept fi)))
      (check-law "gap" (gap coef c*) (* 64 (unit-roundoff 'float32) (max-abs c*))))))

(define (weak-narrow dt g m)
  (define base (normal-case 200 3 5))
  (define (in-dtype x)
    (inexact->exact (rounded dt x)))
  (lcase (format "columns at 2^~a and 2^-~a, y = a + 2^-~a b + 2^-20 c" g g m)
         (for/list ([row (in-list (lcase-X base))])
           (match-define (list a b _) row)
           (list (in-dtype (* (expt 2 g) a)) (in-dtype (* (expt 2 (- g)) b))))
         (for/list ([row (in-list (lcase-X base))])
           (match-define (list a b c) row)
           (in-dtype (+ a (* (expt 2 (- m)) b) (* (expt 2 -20) c))))
         #f))

(test-gpu "a narrow column carrying under 2^-8 (f32) or 2^-18 (f64) of y can be lost when u·κ_α ≥ 1 (known, #104)"
  (for* ([p (in-list '((float32 44 30 9) (float64 100 40 19)))]
         [s (in-list solvers)]
         [fi (in-list '(#t #f))])
    (match-define (list dt g k m) p)
    (define c (weak-narrow dt g m))
    (define alpha (narrow-alpha c k))
    (test-case (format "~a ~a alpha 2^~a ~a ~a" dt (lcase-name c) k s fi)
      (match-define (struct* ridge-solution ([coef (list _ narrow*)]))
        (oracle-ridge (lcase-X c) (lcase-y c) #f alpha fi))
      (match-define (list (list _ narrow) _)
        (exact-fit (ridge-case-fit c dt #:alpha alpha #:solver s #:fit-intercept fi)))
      (check-law "narrow" (abs narrow) (* (expt 2 -30) (abs narrow*))))))
