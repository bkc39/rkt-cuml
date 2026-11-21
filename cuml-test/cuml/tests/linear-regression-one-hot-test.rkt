#lang racket/base

(require (only-in cuml/linear-model ridge-fit)
         (only-in racket/match match match-define match-lambda)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/common-data.rkt" gamma unit-roundoff [on-vector vector])
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/linear-data.rkt"
                  exact-fit
                  gap
                  lstsq
                  max-abs
                  oracle-lstsq
                  outcome
                  prediction-gap
                  solution-coef
                  solution-intercept
                  wobble
                  [on-matrix matrix])
         (only-in "private/linear-refusal-rows.rkt" collinear-refusal not-least-squares)
         (only-in "private/ridge-data.rkt" ridge-result-fit))

(define shapes '((500 3) (500 5) (500 12) (5003 3) (5003 5) (5003 12)))
(define remedy-bound (expt 2 -18))
(define float64-min-norm-bound (expt 2 -40))
(define eig-prediction-bound (expt 2 -12))

(define (prediction-mass rows fit)
  (match-define (list coef b) fit)
  (for/fold ([worst 0]) ([row (in-list rows)])
    (max worst (+ (abs b) (for/sum ([x (in-list row)] [cj (in-list coef)]) (abs (* x cj)))))))

(define (coefficient-rounding fit)
  (match-define (list coef _) fit)
  (gamma (add1 (length coef)) (unit-roundoff 'float32)))

(define (rounded-prediction-bound rows y fit)
  (/ (* (coefficient-rounding fit) (+ (prediction-mass rows fit) (max-abs y))) (max-abs y)))

(define (one-hot-case n k)
  (define (level i)
    (modulo (* 7 i) k))
  (define (spread i)
    (/ (- (modulo (* 13 i) 29) 14) 4))
  (list (for/list ([i (in-range n)])
          (append (for/list ([j (in-range k)])
                    (if (= j (level i)) 1 0))
                  (list (spread i))))
        (for/list ([i (in-range n)])
          (+ (* 3 (level i)) (* 1/2 (spread i)) (/ (- (wobble i) 1/2) 4)))))

(define (min-norm-gap rows y fit)
  (match-define (list coef b) fit)
  (define o (oracle-lstsq rows y #f #t))
  (/ (gap (cons b coef) (cons (solution-intercept o) (solution-coef o))) (max-abs (solution-coef o))))

(test-case "the svd bound admits a range-space error of u_T times the prediction scale"
  (match-define (list rows y) (one-hot-case 500 3))
  (define o (oracle-lstsq rows y #f #t))
  (define coef (solution-coef o))
  (define b (solution-intercept o))
  (define exact (list coef b))
  (define shift
    (* 99/100 (coefficient-rounding exact) (+ (prediction-mass rows exact) (max-abs y))))
  (define moved (list coef (+ b shift)))
  (check-true (< (prediction-gap rows y moved) (rounded-prediction-bound rows y moved))))

(test-gpu "float32 'qr and 'svd-qr refuse one-hot columns with an intercept"
  (for* ([shape (in-list shapes)]
         [algorithm (in-list '(qr svd-qr))])
    (match-define (list rows y) (apply one-hot-case shape))
    (test-case (format "~a ~a" algorithm shape)
      (check-cuml-error 'logic
                        (collinear-refusal algorithm)
                        (lambda () (lstsq rows y 'float32 algorithm))))))

(test-gpu "dropping a level, no intercept, or float64 'auto fits one-hot columns"
  (for ([shape (in-list shapes)])
    (match-define (list rows y) (apply one-hot-case shape))
    (for ([algorithm (in-list '(qr svd-qr))])
      (test-case (format "~a ~a, a level dropped" algorithm shape)
        (define dropped
          (map (match-lambda
                 [(cons _ kept) kept])
               rows))
        (check-true (< (prediction-gap rows y (lstsq dropped y 'float32 algorithm) #:fit-rows dropped)
                       remedy-bound)))
      (test-case (format "~a ~a, no intercept" algorithm shape)
        (define fit (lstsq rows y 'float32 algorithm #:fit-intercept #f))
        (check-true (< (prediction-gap rows y fit) remedy-bound))))
    (test-case (format "float64 auto ~a" shape)
      (check-true (< (min-norm-gap rows y (lstsq rows y 'float64 'auto)) float64-min-norm-bound)))))

(test-gpu "float32 'auto and 'eig fit one-hot columns with an intercept or refuse them as not least squares"
  (for* ([shape (in-list shapes)]
         [algorithm (in-list '(auto eig))])
    (match-define (list rows y) (apply one-hot-case shape))
    (test-case (format "~a ~a" algorithm shape)
      (match (outcome (lambda () (lstsq rows y 'float32 algorithm)))
        [(? string? message) (check-true (regexp-match? not-least-squares message) message)]
        [fit (check-true (< (prediction-gap rows y fit) eig-prediction-bound))]))))

(test-gpu "float32 'svd and 'svd-jacobi fit one-hot columns with an intercept off the minimum norm"
  (for* ([shape (in-list shapes)]
         [algorithm (in-list '(svd svd-jacobi))])
    (match-define (list rows y) (apply one-hot-case shape))
    (define fit (lstsq rows y 'float32 algorithm))
    (test-case (format "~a ~a" algorithm shape)
      (define exact (exact-fit fit))
      (check-true (< (prediction-gap rows y fit) (rounded-prediction-bound rows y exact)))
      (check-true (> (min-norm-gap rows y fit) 1)))))

(test-gpu "float32 ridge at alpha 0 and 1e-12 fits one-hot columns as least squares does"
  (for* ([shape (in-list shapes)]
         [solver (in-list '(auto eig svd))]
         [alpha (in-list '(0.0 1e-12))])
    (match-define (list rows y) (apply one-hot-case shape))
    (test-case (format "~a ~a alpha ~a" solver shape alpha)
      (check-equal?
       (outcome
        (lambda ()
          (~> (ridge-fit (matrix rows 'float32) (vector y 'float32) #:alpha alpha #:solver solver)
              ridge-result-fit
              exact-fit)))
       (match (outcome (lambda () (lstsq rows y 'float32 solver)))
         [(? string? message) (regexp-replace #rx"^linear-regression-fit" message "ridge-fit")]
         [fit fit])))))
