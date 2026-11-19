#lang racket/base

(require (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-result-coef
                  linear-regression-result-intercept
                  ridge-fit
                  ridge-result-coef
                  ridge-result-intercept)
         (only-in racket/match match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/exact-algebra.rkt" dot)
         (only-in "private/linear-data.rkt"
                  gap
                  max-abs
                  oracle-lstsq
                  solution-coef
                  solution-intercept
                  [on-matrix matrix]
                  [on-vector vector])
         (only-in "private/linear-refusal-rows.rkt" collinear-refusal not-least-squares))

(define shapes '((500 3) (500 5) (500 12) (5003 3) (5003 5) (5003 12)))
(define remedy-bound (expt 2 -18))
(define float64-min-norm-bound (expt 2 -40))
(define eig-prediction-bound (expt 2 -12))
(define svd-prediction-bound (expt 2 -10))

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
          (+ (* 3 (level i)) (* 1/2 (spread i)) (/ (- (modulo (* 31 i) 17) 8) 64)))))

(define (fitted fit coef-of intercept-of)
  (cons (inexact->exact (intercept-of fit)) (map inexact->exact (device-vector->list (coef-of fit)))))

(define (lstsq rows y dt algorithm #:fit-intercept [fi #t])
  (fitted
   (linear-regression-fit (matrix rows dt) (vector y dt) #:algorithm algorithm #:fit-intercept fi)
   linear-regression-result-coef
   linear-regression-result-intercept))

(define (prediction-gap rows y b+c #:fit-rows [fit-rows rows])
  (define o (oracle-lstsq rows y #f #t))
  (define (predict on b c)
    (for/list ([row (in-list on)])
      (+ b (dot row c))))
  (/ (gap (predict fit-rows (car b+c) (cdr b+c))
          (predict rows (solution-intercept o) (solution-coef o)))
     (max-abs y)))

(define (min-norm-gap rows y b+c)
  (define o (oracle-lstsq rows y #f #t))
  (/ (gap b+c (cons (solution-intercept o) (solution-coef o))) (max-abs (solution-coef o))))

(define (outcome thunk)
  (with-handlers ([exn:fail? exn-message])
    (thunk)))

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
        (define dropped (map cdr rows))
        (check-true (< (prediction-gap rows y (lstsq dropped y 'float32 algorithm) #:fit-rows dropped)
                       remedy-bound)))
      (test-case (format "~a ~a, no intercept" algorithm shape)
        (define b+c (lstsq rows y 'float32 algorithm #:fit-intercept #f))
        (check-true (< (prediction-gap rows y b+c) remedy-bound))))
    (test-case (format "float64 auto ~a" shape)
      (check-true (< (min-norm-gap rows y (lstsq rows y 'float64 'auto)) float64-min-norm-bound)))))

(test-gpu "float32 'auto and 'eig fit one-hot columns with an intercept or refuse them as not least squares"
  (for* ([shape (in-list shapes)]
         [algorithm (in-list '(auto eig))])
    (match-define (list rows y) (apply one-hot-case shape))
    (test-case (format "~a ~a" algorithm shape)
      (match (outcome (lambda () (lstsq rows y 'float32 algorithm)))
        [(? string? message) (check-true (regexp-match? not-least-squares message) message)]
        [b+c (check-true (< (prediction-gap rows y b+c) eig-prediction-bound))]))))

(test-gpu "float32 'svd and 'svd-jacobi fit one-hot columns with an intercept off the minimum norm"
  (for* ([shape (in-list shapes)]
         [algorithm (in-list '(svd svd-jacobi))])
    (match-define (list rows y) (apply one-hot-case shape))
    (define b+c (lstsq rows y 'float32 algorithm))
    (test-case (format "~a ~a" algorithm shape)
      (check-true (< (prediction-gap rows y b+c) svd-prediction-bound))
      (check-true (> (min-norm-gap rows y b+c) 1)))))

(test-gpu "float32 ridge at alpha 0 and 1e-12 fits one-hot columns as least squares does"
  (for* ([shape (in-list shapes)]
         [solver (in-list '(auto eig svd))]
         [alpha (in-list '(0.0 1e-12))])
    (match-define (list rows y) (apply one-hot-case shape))
    (test-case (format "~a ~a alpha ~a" solver shape alpha)
      (check-equal?
       (outcome
        (lambda ()
          (fitted (ridge-fit (matrix rows 'float32) (vector y 'float32) #:alpha alpha #:solver solver)
                  ridge-result-coef
                  ridge-result-intercept)))
       (match (outcome (lambda () (lstsq rows y 'float32 solver)))
         [(? string? message) (regexp-replace #rx"^linear-regression-fit" message "ridge-fit")]
         [b+c b+c])))))
