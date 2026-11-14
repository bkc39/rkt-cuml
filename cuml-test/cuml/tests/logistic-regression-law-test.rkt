#lang racket/base

(require (only-in cuml/linear-model
                  logistic-regression-decision-function
                  logistic-regression-fit
                  logistic-regression-result-coef
                  logistic-regression-result-intercept)
         (only-in racket/list first remove-duplicates rest)
         (only-in racket/match match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" by-dtype dtypes on-matrix on-vector unit-roundoff)
         (only-in "private/logistic-data.rkt"
                  blob-case
                  case-fit
                  case-y
                  fit-flat
                  fit-parts
                  lr-case
                  lr-case-X
                  lr-case-name
                  lr-case-w
                  lr-case-y
                  lr-fit
                  lr-fit-coef
                  relative-gap
                  scaled-case
                  scaled-parts
                  tight-max-iter
                  tight-tol
                  with-column))

(define binary (blob-case "binary" 80 3 2 11 #:weighted? #t))
(define three (blob-case "three classes" 90 3 3 12))

(define (zeroed-every k)
  (for/list ([w (in-list (lr-case-w binary))]
             [i (in-naturals)])
    (if (zero? (modulo i k)) 0 w)))

(define (penalty-scale a penalty)
  (case penalty
    [(l2) (expt 4 (- a))]
    [(l1) (expt 2 (- a))]
    [else 1]))

(define (count-of ys c)
  (for/sum ([y (in-list ys)]) (if (= y c) 1 0)))

(test-gpu "a constant or zero column gets coef 0 and leaves the rest bitwise"
  (for* ([c (in-list (list binary three))]
         [dt (in-list dtypes)]
         [column (in-list '((3.0 #t) (0.0 #f)))])
    (match-define (list value fi) column)
    (test-case (format "~a ~a ~a" (lr-case-name c) dt column)
      (define base (fit-parts (case-fit c dt #:fit-intercept fi)))
      (define wide (fit-parts (case-fit (with-column c value) dt #:fit-intercept fi)))
      (define firsts (map first (lr-fit-coef wide)))
      (define rests (map rest (lr-fit-coef wide)))
      (check-true (andmap zero? firsts))
      (check-equal? (struct-copy lr-fit wide [coef rests]) base))))

(test-gpu "rows of weight 0 fit as deleted rows, bitwise"
  (define zeroed (zeroed-every 3))
  (define kept
    (for/list ([w (in-list zeroed)])
      (positive? w)))
  (define (keep xs)
    (for/list ([x (in-list xs)]
               [k (in-list kept)]
               #:when k)
      x))
  (for ([dt (in-list dtypes)])
    (define with-zeros (lr-case "zeros" (lr-case-X binary) (lr-case-y binary) zeroed))
    (define deleted
      (lr-case "deleted" (keep (lr-case-X binary)) (keep (lr-case-y binary)) (keep zeroed)))
    (check-equal? (fit-parts (case-fit with-zeros dt)) (fit-parts (case-fit deleted dt)))))

(test-gpu "weights times 2^a fit as C times 2^a, bitwise, and leave 'none unchanged"
  (for* ([dt (in-list dtypes)]
         [a (in-list '(-20 3 40))])
    (test-case (format "~a ~a" dt a)
      (define factor (expt 2 a))
      (define scaled
        (lr-case "scaled"
                 (lr-case-X binary)
                 (lr-case-y binary)
                 (map (lambda (w) (* w factor)) (lr-case-w binary))))
      (check-equal? (fit-parts (case-fit scaled dt))
                    (fit-parts (case-fit binary dt #:C (exact->inexact factor))))
      (check-equal? (fit-parts (case-fit scaled dt #:penalty 'none))
                    (fit-parts (case-fit binary dt #:penalty 'none)))
      (define uniform
        (lr-case "uniform"
                 (lr-case-X three)
                 (lr-case-y three)
                 (map (lambda (_) factor) (lr-case-y three))))
      (check-equal? (fit-parts (case-fit uniform dt))
                    (fit-parts (case-fit three dt #:C (exact->inexact factor)))))))

(test-gpu "X times 2^a scales coef by 2^-a bitwise when both fits are staged"
  (for* ([dt (in-list dtypes)]
         [a (in-list '(12 -12 20 -20 40 -40))]
         [penalty (in-list '(l2 l1 none))])
    (test-case (format "~a ~a ~a" dt a penalty)
      (define base (scaled-case binary (expt 2 (if (positive? a) 10 -10))))
      (define C (penalty-scale a penalty))
      (define plain (fit-parts (case-fit base dt #:penalty penalty)))
      (define moved
        (fit-parts
         (case-fit (scaled-case base (expt 2 a)) dt #:penalty penalty #:C (exact->inexact C))))
      (check-equal? moved (scaled-parts plain (expt 2.0 (- a)))))))

(test-gpu "balanced class weights equal the explicit mapping W/(k W_c), bitwise"
  (define ys (lr-case-y binary))
  (define classes (sort (remove-duplicates ys) <))
  (define n (length ys))
  (define mapping
    (for/hash ([c (in-list classes)])
      (values c (exact->inexact (/ n (* (length classes) (count-of ys c)))))))
  (define unweighted (lr-case "unweighted" (lr-case-X binary) ys #f))
  (for ([dt (in-list dtypes)])
    (check-equal? (fit-parts (case-fit unweighted dt #:class-weight 'balanced))
                  (fit-parts (case-fit unweighted dt #:class-weight mapping)))))

(define intercept-ulps 4)

(test-gpu "no kept columns give the closed-form intercept"
  (define ys '(0 0 1 1 1 2 2 2 2 2))
  (for* ([dt (in-list dtypes)]
         [labels (in-list (list (map (lambda (y) (min y 1)) ys) ys))])
    (define X (map (lambda (_) '(2.5 -1.0)) labels))
    (define r (logistic-regression-fit (on-matrix X dt) (on-vector labels 'int32)))
    (define counts
      (for/list ([c (in-list (sort (remove-duplicates labels) <))])
        (count-of labels c)))
    (define expected
      (match counts
        [(list negative positive) (list (log (/ positive negative)))]
        [_
         (define logs (map log counts))
         (map (lambda (l) (- l (/ (apply + logs) (length logs)))) logs)]))
    (define u (unit-roundoff dt))
    (check-true (andmap (lambda (row) (andmap zero? row))
                        (device-matrix->list* (logistic-regression-result-coef r))))
    (for ([b (in-list (device-vector->list (logistic-regression-result-intercept r)))]
          [e (in-list expected)])
      (check-true (<= (abs (- b e)) (* intercept-ulps u (max 1 (abs e))))))))

(test-gpu "the fit leaves X, y and the weights as they were"
  (define zeroed (zeroed-every 4))
  (define staged (scaled-case binary (expt 2 20)))
  (for ([dt (in-list dtypes)])
    (define X (on-matrix (lr-case-X staged) dt))
    (define y (case-y binary))
    (define w (on-vector zeroed dt))
    (define before (list (device-matrix->list* X) (device-vector->list y) (device-vector->list w)))
    (logistic-regression-fit X y #:sample-weight w)
    (check-equal? (list (device-matrix->list* X) (device-vector->list y) (device-vector->list w))
                  before)))

(test-gpu "the decision is X coef^T + b to rounding"
  (for* ([c (in-list (list binary three))]
         [dt (in-list dtypes)])
    (test-case (format "~a ~a" (lr-case-name c) dt)
      (define r (case-fit c dt))
      (define coef
        (map (lambda (row) (map inexact->exact row))
             (device-matrix->list* (logistic-regression-result-coef r))))
      (define b (map inexact->exact (device-vector->list (logistic-regression-result-intercept r))))
      (define u (inexact->exact (unit-roundoff dt)))
      (define z
        (device-matrix->list* (logistic-regression-decision-function r (on-matrix (lr-case-X c) dt))))
      (for* ([(x zi) (in-parallel (in-list (lr-case-X c)) (in-list z))]
             [(w bc zc) (in-parallel (in-list coef) (in-list b) (in-list zi))])
        (define exact (+ bc (for/sum ([xj (in-list x)] [wj (in-list w)]) (* xj wj))))
        (define size (+ (abs bc) (for/sum ([xj (in-list x)] [wj (in-list w)]) (abs (* xj wj)))))
        (check-true (<= (abs (- (inexact->exact zc) exact)) (* (add1 (length x)) u size)))))))

(define (repeated c times)
  (define (spread xs)
    (for*/list ([(x y) (in-parallel (in-list xs) (in-list (lr-case-y c)))]
                [_ (in-range (hash-ref times y))])
      x))
  (lr-case (lr-case-name c) (spread (lr-case-X c)) (spread (lr-case-y c)) #f))

(define repeated-gap-bound 1e-6)

(test-gpu "integer class weights fit as repeated rows"
  (define unweighted-binary (lr-case "binary" (lr-case-X binary) (lr-case-y binary) #f))
  (for ([c (in-list (list unweighted-binary three))]
        [times (in-list (list (hash 0 2 1 3) (hash 0 2 1 2 2 2)))])
    (test-case (lr-case-name c)
      (define tol (tight-tol 'float64))
      (define weighted
        (fit-parts (case-fit c 'float64 #:class-weight times #:tol tol #:max-iter tight-max-iter)))
      (define spread
        (fit-parts (case-fit (repeated c times) 'float64 #:tol tol #:max-iter tight-max-iter)))
      (define gap (relative-gap (fit-flat weighted) (fit-flat spread)))
      (check-true (<= gap repeated-gap-bound) (format "gap ~a" gap)))))

(define (window-scale-gap-bound dt)
  (by-dtype dt (min (* 8 2e-3) 8e-2) (min (* 8 1e-6) 3e-2)))

(test-gpu "X times 2^a inside the staging window scales coef by 2^-a to the solver's tolerance"
  (for* ([dt (in-list dtypes)]
         [a (in-list '(3 -3))]
         [penalty (in-list '(l2 l1 none))])
    (test-case (format "~a ~a ~a" dt a penalty)
      (define C (penalty-scale a penalty))
      (define tight (tight-tol dt))
      (define plain
        (fit-parts (case-fit binary dt #:penalty penalty #:tol tight #:max-iter tight-max-iter)))
      (define moved
        (fit-parts (case-fit (scaled-case binary (expt 2 a))
                             dt
                             #:penalty penalty
                             #:C (exact->inexact C)
                             #:tol tight
                             #:max-iter tight-max-iter)))
      (define gap (relative-gap (fit-flat moved) (fit-flat (scaled-parts plain (expt 2.0 (- a))))))
      (check-true (<= gap (window-scale-gap-bound dt)) (format "gap ~a" gap)))))
