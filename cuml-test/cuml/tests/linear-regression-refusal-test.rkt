#lang racket/base

(require (only-in cuml fit linear-regression predict)
         (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-predict
                  linear-regression-result)
         (only-in racket/list make-list range)
         (only-in rackunit check-equal? test-case)
         (only-in raft device-matrix->list* device-vector->list list*->device-matrix list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "private/linear-data.rkt" algorithms dtypes)
         (only-in "private/refusal-rows.rkt" check-row row row-id)
         (only-in "private/scaler-data.rkt" strided-vector))

(define grid '((1 0 2) (0 3 1) (4 1 0) (2 2 5) (3 1 1)))
(define targets '(1 2 3 4 5))

(define (matrix rows [dt 'float64])
  (list*->device-matrix rows #:dtype dt))

(define (vector xs [dt 'float64])
  (list->device-vector xs #:dtype dt))

(define (with-cell rows i j value)
  (for/list ([row (in-list rows)] [r (in-naturals)])
    (for/list ([x (in-list row)] [c (in-naturals)])
      (if (and (= r i) (= c j)) value x))))

(define (with-element xs i value)
  (for/list ([x (in-list xs)] [k (in-naturals)])
    (if (= k i) value x)))

(define (fit-rows go)
  (define X (matrix grid))
  (define y (vector targets))
  (list
   (row "V1 dtype" "X: expected float32 or float64, got int32" (lambda () (go (blank-matrix 5 3 'int32) y)))
   (row "V1 rank" "X: expected rank 2, got rank 1" (lambda () (go y y)))
   (row "V1 layout"
        "X: expected row-major, got col-major 4x3"
        (lambda () (go (blank-matrix 4 3 'float32 #:layout 'col-major) (blank-vector 4 'float32))))
   (row "V2 dtype" "y: expected float32, got float64" (lambda () (go (matrix grid 'float32) y)))
   (row "V2 rank" "y: expected rank 1, got rank 2" (lambda () (go X X)))
   (row "V2 length"
        "y: expected 10 elements, got 9"
        (lambda () (go (matrix (append grid grid)) (vector (range 9)))))
   (row "V2 stride" "y: expected contiguous elements, got stride 2" (lambda () (go X (strided-vector 5))))
   (row "V3 dtype"
        "sample-weight: expected float32, got float64"
        (lambda () (go (matrix grid 'float32) (vector targets 'float32) #:sample-weight y)))
   (row "V3 length"
        "sample-weight: expected 5 elements, got 4"
        (lambda () (go X y #:sample-weight (vector '(1 1 1 1)))))
   (row "V4" "X: expected at least 2 rows, got 1" (lambda () (go (matrix '((1 2 3))) (vector '(1)))))
   (row "V5" "X: expected at least 1 column, got 0" (lambda () (go (blank-matrix 5 0 'float64) y)))
   (row "V8 algorithm"
        "algorithm: expected auto, eig, svd, svd-jacobi, qr or svd-qr"
        (lambda () (go X y #:algorithm 'lu)))
   (row "V8 fit-intercept" "fit-intercept: expected #t or #f" (lambda () (go X y #:fit-intercept 'yes)))
   (row "V9"
        "X: row 3, column 1: expected a finite value, got nan"
        (lambda () (go (matrix (with-cell grid 3 1 +nan.0)) y)))
   (row "V10 y"
        "y: element 4: expected a finite value, got inf"
        (lambda () (go X (vector (with-element targets 4 +inf.0)))))
   (row "V10 sample-weight"
        "sample-weight: element 0: expected a finite value, got nan"
        (lambda () (go X y #:sample-weight (vector '(+nan.0 1 1 1 1)))))
   (row "V11"
        "sample-weight: element 1: expected at least 0, got -1"
        (lambda () (go X y #:sample-weight (vector '(1 -1 1 1 1)))))
   (row "V12"
        "sample-weight: expected a positive sum, got 0"
        (lambda () (go X y #:sample-weight (vector (make-list 5 0)))))
   (row "V14"
        "X: column 2: the coefficient exceeds the float32 range"
        (lambda ()
          (go (matrix (for/list ([r (in-list grid)])
                        (list (car r) (cadr r) (* (caddr r) (expt 2 -100))))
                      'float32)
              (vector (for/list ([r (in-list grid)] [t (in-list targets)])
                        (* (expt 2 60) (+ t (caddr r))))
                      'float32))))))

(define (predict-rows fitted go)
  (define X (matrix grid))
  (list (row "W1 dtype" "X: expected float32 or float64, got int32" (lambda () (go fitted (blank-matrix 5 3 'int32))))
        (row "W1 rank" "X: expected rank 2, got rank 1" (lambda () (go fitted (vector targets))))
        (row "W1 layout"
             "X: expected row-major, got col-major 4x3"
             (lambda () (go fitted (blank-matrix 4 3 'float64 #:layout 'col-major))))
        (row "W3" "X: expected float64, got float32" (lambda () (go fitted (matrix grid 'float32))))
        (row "W4" "X: expected at least 1 row, got 0" (lambda () (go fitted (blank-matrix 0 3 'float64))))
        (row "W5" "X: expected 3 columns, got 2" (lambda () (go fitted (matrix '((1 2) (3 4))))))
        (row "W7 X"
             "X: row 3, column 1: expected a finite value, got nan"
             (lambda () (go fitted (matrix (with-cell grid 3 1 +nan.0)))))))

(define (result-rows go)
  (define X (matrix grid))
  (list (row "W2 rank"
             "coef: expected rank 1, got rank 2"
             (lambda () (go (linear-regression-result X 0.0) X)))
        (row "W2 empty"
             "coef: expected at least 1 element, got 0"
             (lambda () (go (linear-regression-result (blank-vector 0 'float64) 0.0) X)))
        (row "W7 coef"
             "coef: element 0: expected a finite value, got nan"
             (lambda () (go (linear-regression-result (vector '(+nan.0 2 3)) 0.0) X)))
        (row "W7 intercept"
             "intercept: expected a finite value, got inf"
             (lambda () (go (linear-regression-result (vector '(1 2 3)) +inf.0) X)))
        (row "W8"
             "X: row 0: the prediction exceeds the float32 range"
             (lambda ()
               (go (linear-regression-result (vector '(1 1 1) 'float32) 0.0)
                   (matrix '((3e38 3e38 3e38) (1 1 1)) 'float32))))))

(define (direct-fit X y . options)
  (apply linear-regression-fit X y options))

(define (estimator-fit X y #:sample-weight [w #f] #:algorithm [a 'auto] #:fit-intercept [fi #t])
  (fit (linear-regression #:algorithm a #:fit-intercept fi) X y #:sample-weight w))

(define (keyword-fit proc)
  (make-keyword-procedure (lambda (kws vals X y) (keyword-apply proc kws vals X y '()))))

(test-gpu "R1, V1-V14: linear-regression-fit refuses each row before it solves"
  (for ([r (in-list (fit-rows (keyword-fit direct-fit)))])
    (test-case (row-id r)
      (check-row "linear-regression-fit" r))))

(test-gpu "R1, V1-V14: the estimator's fit names the generic"
  (for ([r (in-list (fit-rows (keyword-fit estimator-fit)))])
    (test-case (row-id r)
      (check-row "fit" r))))

(test-gpu "R1, W1-W8: linear-regression-predict refuses each row"
  (define fitted (linear-regression-fit (matrix grid) (vector targets)))
  (for ([r (in-list (append (predict-rows fitted linear-regression-predict)
                            (result-rows linear-regression-predict)))])
    (test-case (row-id r)
      (check-row "linear-regression-predict" r))))

(test-gpu "R1, W1-W7: the estimator's predict names the generic"
  (define fitted (fit (linear-regression) (matrix grid) (vector targets)))
  (for ([r (in-list (predict-rows fitted predict))])
    (test-case (row-id r)
      (check-row "predict" r))))

(test-case "the estimator refuses a missing y and predict before fit"
  (check-cuml-error 'logic
                    "fit: linear-regression needs y"
                    (lambda () (fit (linear-regression) (matrix grid))))
  (check-cuml-error 'logic
                    "predict: linear-regression is not fitted"
                    (lambda () (predict (linear-regression) (matrix grid)))))

(define v13
  #px"^linear-regression-fit: X: the (eig|svd|qr|svd-qr) solve is not a least-squares solution \\(residual cosine \\S+ > 2\\^-(8|18)\\); X may have collinear columns$")

(define (collinear-rows n)
  (for/list ([i (in-range n)])
    (define a (/ (modulo (* 37 i) 101) 16))
    (define b (/ (modulo (* 53 i) 97) 32))
    (list a b (+ a b))))

(define (one-hot-rows n k)
  (for/list ([i (in-range n)])
    (for/list ([j (in-range k)])
      (if (= j (modulo i k)) 1 0))))

(define (vander-rows n degree)
  (for/list ([i (in-range n)])
    (define t (/ i (sub1 n)))
    (for/list ([p (in-range 1 (add1 degree))])
      (expt t p))))

(define (noisy-targets rows)
  (for/list ([r (in-list rows)] [i (in-naturals)])
    (+ (apply + r) (/ (modulo (* 31 i) 17) 64))))

(test-gpu "V13: a solve that is not a least-squares solution is refused"
  (for ([fixture (in-list (list (list 'eig 'float32 (collinear-rows 400))
                                (list 'qr 'float32 (one-hot-rows 400 4))
                                (list 'qr 'float64 (vander-rows 400 9))))])
    (define-values (algorithm dt rows) (apply values fixture))
    (test-case (format "~a ~a" algorithm dt)
      (check-cuml-error 'logic
                        v13
                        (lambda ()
                          (linear-regression-fit (matrix rows dt)
                                                 (vector (noisy-targets rows) dt)
                                                 #:algorithm algorithm))))))

(test-gpu "R2: X, y and w are unchanged after every fit"
  (for* ([dt (in-list dtypes)]
         [algorithm (in-list algorithms)]
         [w (in-list (list #f '(1 0 2 3 1/2)))]
         [fi (in-list '(#t #f))])
    (define X (matrix grid dt))
    (define y (vector targets dt))
    (define W (and w (vector w dt)))
    (define (snapshot)
      (list (device-matrix->list* X) (device-vector->list y) (and W (device-vector->list W))))
    (define before (snapshot))
    (linear-regression-fit X y #:sample-weight W #:algorithm algorithm #:fit-intercept fi)
    (test-case (format "~a ~a ~a ~a" dt algorithm (and w #t) fi)
      (check-equal? (snapshot) before))))
