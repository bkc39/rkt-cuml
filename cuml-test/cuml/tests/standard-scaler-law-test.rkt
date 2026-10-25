#lang racket/base

(require (only-in cuml/preprocessing
                  standard-scaler-fit
                  standard-scaler-inverse-transform
                  standard-scaler-result-mean
                  standard-scaler-result-scale
                  standard-scaler-result-var
                  standard-scaler-transform)
         (only-in cuml/preprocessing/private/standard-scaler current-scratch-elements)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-eqv? check-false check-true test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/scaler-data.rkt"
                  bits-equal?
                  constant-rows
                  dyadic-rows
                  host-inverse
                  host-transform
                  normal-rows
                  round-to
                  rows->matrix
                  standardisation-errors
                  ulp
                  unit-roundoff))

(define flag-pairs '((#t #t) (#t #f) (#f #t) (#f #f)))

(define (fit-with X flags)
  (match-define (list with-mean with-std) flags)
  (standard-scaler-fit X #:with-mean with-mean #:with-std with-std))

(define (statistics result)
  (for/list ([v (list (standard-scaler-result-mean result)
                      (standard-scaler-result-var result)
                      (standard-scaler-result-scale result))])
    (and v (device-vector->list v))))

(test-gpu "L2, L3: transform and inverse equal the host oracles bitwise, every dtype, flag pair and size"
  (for* ([dt (in-list '(float32 float64))]
         [n (in-list '(1 7 256 4099))]
         [d (in-list '(1 13 64))])
    (define rows (normal-rows (+ (* 31 n) d) n d dt))
    (define X (rows->matrix rows dt))
    (define host (device-matrix->list* X))
    (for ([flags (in-list flag-pairs)])
      (test-case (format "~a ~ax~a ~a" dt n d flags)
        (define fitted (fit-with X flags))
        (define T (standard-scaler-transform fitted X))
        (define t-host (device-matrix->list* T))
        (check-true (bits-equal? t-host (host-transform fitted host dt)) "transform")
        (check-true (bits-equal? (device-matrix->list* (standard-scaler-inverse-transform fitted T))
                                 (host-inverse fitted t-host dt))
                    "inverse")))))

(test-gpu "L4: dyadic columns give exact statistics, an exact transform and an exact inverse"
  (for* ([dt (in-list '(float32 float64))]
         [k (in-list '(-3 0 5))]
         [c (in-list '(0 12 -7.5))])
    (test-case (format "~a k=~a c=~a" dt k c)
      (define X (rows->matrix (dyadic-rows c k) dt))
      (define fitted (standard-scaler-fit X))
      (check-equal? (statistics fitted)
                    (map (lambda (v) (list (exact->inexact v))) (list c (expt 4 k) (expt 2 k))))
      (define T (standard-scaler-transform fitted X))
      (check-equal? (device-matrix->list* T)
                    (for/list ([i (in-range 8)])
                      (list (if (even? i) 1.0 -1.0))))
      (check-true (bits-equal? (device-matrix->list* (standard-scaler-inverse-transform fitted T))
                               (device-matrix->list* X))))))

(test-gpu "L5: the round trip is within the first-order rounding bound"
  (for* ([dt (in-list '(float32 float64))]
         [size (in-list '((7 13) (256 64) (4099 13)))])
    (match-define (list n d) size)
    (define X (rows->matrix (normal-rows (+ n (* 3 d)) n d dt) dt))
    (define fitted (standard-scaler-fit X))
    (define u (unit-roundoff dt))
    (define means (map (round-to dt) (device-vector->list (standard-scaler-result-mean fitted))))
    (define back
      (device-matrix->list* (standard-scaler-inverse-transform fitted
                                                               (standard-scaler-transform fitted X))))
    (define worst
      (for*/fold ([w 0.0])
                 ([(row back-row) (in-parallel (device-matrix->list* X) back)]
                  [(x b m) (in-parallel row back-row means)])
        (max w (/ (abs (- b x)) (* 1.01 u (+ (* 3 (abs (- x m))) (abs x)))))))
    (check-true (<= worst 1.0) (format "~a ~a: worst ratio ~a" dt size worst))))

(define (columns rows)
  (apply map list rows))

(test-gpu "L6: standardised columns have mean 0 and sd 1, within the rounding of the statistics"
  (for* ([dt (in-list '(float32 float64))]
         [size (in-list '((7 13) (256 64) (4099 13)))])
    (match-define (list n d) size)
    (define X (rows->matrix (normal-rows (+ 5 n d) n d dt) dt))
    (define fitted (standard-scaler-fit X))
    (define T (standard-scaler-transform fitted X))
    (for ([x-column (in-list (columns (device-matrix->list* X)))]
          [t-column (in-list (columns (device-matrix->list* T)))]
          [mean (in-list (device-vector->list (standard-scaler-result-mean fitted)))]
          [scale (in-list (device-vector->list (standard-scaler-result-scale fitted)))]
          [j (in-naturals)])
      (for ([ratio (in-list (standardisation-errors x-column t-column mean scale dt))]
            [bound (in-list '("mean against the statistics" "mean" "sd"))])
        (check-true (<= ratio 1)
                    (format "~a ~a column ~a, ~a: ~a of its bound" dt size j bound ratio))))))

(test-gpu "L7: a constant column scales by exactly 1 and centres to (nearly) 0"
  (for* ([dt (in-list '(float32 float64))]
         [value (in-list '(3.3 0.1 -1234.5678 1e4))]
         [n (in-list '(1 2 9 1000))])
    (test-case (format "~a ~a x~a" dt value n)
      (define rounded ((round-to dt) value))
      (define X (rows->matrix (constant-rows n rounded) dt))
      (define fitted (standard-scaler-fit X))
      (match-define (list _ (list var) (list scale)) (statistics fitted))
      (check-equal? scale 1.0)
      (check-eqv? var 0.0)
      (define ts (map car (device-matrix->list* (standard-scaler-transform fitted X))))
      (cond
        [(or (eq? dt 'float32) (= n 1)) (check-equal? ts (make-list n 0.0))]
        [else
         (for ([t (in-list ts)])
           (check-true (<= (abs t) (ulp rounded)) (format "t ~a" t)))]))))

(test-gpu "L8: the near-constant mask is sklearn's"
  (define base 1e8)
  (define tiny (rows->matrix (list (list base) (list base) (list (+ base (ulp base)))) 'float64))
  (check-equal? (device-vector->list (standard-scaler-result-scale (standard-scaler-fit tiny)))
                '(1.0))
  (define spread (rows->matrix (cons (list (+ 1e6 1e-4)) (constant-rows 999 1e6)) 'float64))
  (check-false
   (equal? (device-vector->list (standard-scaler-result-scale (standard-scaler-fit spread))) '(1.0))))

(test-gpu "L9: the flags decide what is kept and what is applied"
  (for ([dt (in-list '(float32 float64))])
    (define X (rows->matrix (normal-rows 17 50 6 dt) dt))
    (define host (device-matrix->list* X))
    (define full (standard-scaler-fit X))
    (define none (fit-with X '(#f #f)))
    (check-equal? (statistics none) '(#f #f #f))
    (check-true (bits-equal? (device-matrix->list* (standard-scaler-transform none X)) host))
    (check-true (bits-equal? (device-matrix->list* (standard-scaler-inverse-transform none X)) host))
    (define scaled (fit-with X '(#f #t)))
    (check-true (bits-equal? (statistics scaled) (statistics full)))
    (define s (map (round-to dt) (device-vector->list (standard-scaler-result-scale scaled))))
    (define m (map (round-to dt) (device-vector->list (standard-scaler-result-mean full))))
    (define narrow (round-to dt))
    (check-true (bits-equal? (device-matrix->list* (standard-scaler-transform scaled X))
                             (for/list ([row (in-list host)])
                               (for/list ([x (in-list row)]
                                          [k (in-list s)])
                                 (narrow (/ x k))))))
    (define centred (fit-with X '(#t #f)))
    (check-equal? (cdr (statistics centred)) '(#f #f))
    (check-true (bits-equal? (car (statistics centred)) (car (statistics full))))
    (check-true (bits-equal? (device-matrix->list* (standard-scaler-transform centred X))
                             (for/list ([row (in-list host)])
                               (for/list ([x (in-list row)]
                                          [k (in-list m)])
                                 (narrow (- x k))))))))

(test-gpu "L10: fits are deterministic, whatever the scratch"
  (for ([dt (in-list '(float32 float64))])
    (define n 1000)
    (define d 24)
    (define X (rows->matrix (normal-rows 99 n d dt) dt))
    (define once (statistics (standard-scaler-fit X)))
    (check-true (bits-equal? (statistics (standard-scaler-fit X)) once) "a second fit")
    (for ([elements (in-list (list n (* 3 n) (sub1 (* n d)) (expt 2 26)))])
      (parameterize ([current-scratch-elements elements])
        (check-true (bits-equal? (statistics (standard-scaler-fit X)) once)
                    (format "~a scratch ~a" dt elements))))))

(test-gpu "L12: a fit transforms either dtype, rounding its statistics to X's"
  (define rows (normal-rows 41 64 5 'float32))
  (define X32 (rows->matrix rows 'float32))
  (define X64 (rows->matrix rows 'float64))
  (define fit32 (standard-scaler-fit X32))
  (define fit64 (standard-scaler-fit X64))
  (check-true (bits-equal? (statistics fit32) (statistics fit64)) "float32 values fit as float64")
  (for ([fitted (list fit32 fit64)])
    (check-true (bits-equal? (device-matrix->list* (standard-scaler-transform fitted X64))
                             (host-transform fitted (device-matrix->list* X64) 'float64)))
    (check-true (bits-equal? (device-matrix->list* (standard-scaler-transform fitted X32))
                             (host-transform fitted (device-matrix->list* X32) 'float32)))))
