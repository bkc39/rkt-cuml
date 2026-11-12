#lang racket/base

(require (only-in cuml fit linear-regression predict)
         (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-predict
                  linear-regression-result)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" uniform-source)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "private/linear-data.rkt"
                  algorithms
                  dtypes
                  dyadic
                  [on-matrix matrix]
                  [on-vector vector])
         (only-in "private/linear-refusal-rows.rkt"
                  grid
                  overflow-X
                  overflow-y
                  shared-fit-rows
                  targets
                  with-cell)
         (only-in "private/refusal-rows.rkt" check-row row row-id)
         (only-in "private/scaler-data.rkt" strided-vector))

(define (fit-rows go)
  (define X (matrix grid))
  (define y (vector targets))
  (append
   (list (row "X layout"
              "X: expected row-major, got col-major 4x3"
              (lambda ()
                (go (blank-matrix 4 3 'float32 #:layout 'col-major) (blank-vector 4 'float32))))
         (row "y rank" "y: expected rank 1, got rank 2" (lambda () (go X X)))
         (row "y stride"
              "y: expected contiguous elements, got stride 2"
              (lambda () (go X (strided-vector 5))))
         (row "weight dtype"
              "sample-weight: expected float32, got float64"
              (lambda () (go (matrix grid 'float32) (vector targets 'float32) #:sample-weight y)))
         (row "unknown algorithm"
              "algorithm: expected auto, eig, svd, svd-jacobi, qr or svd-qr"
              (lambda () (go X y #:algorithm 'lu)))
         (row "non-boolean fit-intercept"
              "fit-intercept: expected #t or #f, got 'yes"
              (lambda () (go X y #:fit-intercept 'yes)))
         (row "non-boolean fit-intercept before the X views"
              "fit-intercept: expected #t or #f, got 'yes"
              (lambda () (go (blank-matrix 5 3 'int32) y #:fit-intercept 'yes)))
         (row "non-finite weight"
              "sample-weight: element 0: expected a finite value, got nan"
              (lambda () (go X y #:sample-weight (vector '(+nan.0 1 1 1 1)))))
         (row "coefficient overflow"
              "X: column 2: the coefficient exceeds the float32 range"
              (lambda () (go (overflow-X) (overflow-y)))))
   (shared-fit-rows go)))

(define (predict-rows fitted go)
  (list (row "X dtype"
             "X: expected float32 or float64, got int32"
             (lambda () (go fitted (blank-matrix 5 3 'int32))))
        (row "X rank" "X: expected rank 2, got rank 1" (lambda () (go fitted (vector targets))))
        (row "X layout"
             "X: expected row-major, got col-major 4x3"
             (lambda () (go fitted (blank-matrix 4 3 'float64 #:layout 'col-major))))
        (row "X dtype unlike coef"
             "X: expected float64, got float32"
             (lambda () (go fitted (matrix grid 'float32))))
        (row "no rows"
             "X: expected at least 1 row, got 0"
             (lambda () (go fitted (blank-matrix 0 3 'float64))))
        (row "X width unlike coef"
             "X: expected 3 columns, got 2"
             (lambda () (go fitted (matrix '((1 2) (3 4))))))
        (row "non-finite X"
             "X: row 3, column 1: expected a finite value, got nan"
             (lambda () (go fitted (matrix (with-cell grid 3 1 +nan.0)))))))

(define (result-rows go)
  (define X (matrix grid))
  (list
   (row "coef rank"
        "coef: expected rank 1, got rank 2"
        (lambda () (go (linear-regression-result X 0.0) X)))
   (row "X rank before the dtype mismatch"
        "X: expected rank 2, got rank 1"
        (lambda () (go (linear-regression-result (vector '(1 2)) 0.0) (vector '(1 2 3) 'float32))))
   (row "coef rank before the dtype mismatch"
        "coef: expected rank 1, got rank 2"
        (lambda ()
          (go (linear-regression-result (matrix '((1 2) (3 4))) 0.0)
              (matrix '((1 2) (3 4)) 'float32))))
   (row "empty coef"
        "coef: expected at least 1 element, got 0"
        (lambda () (go (linear-regression-result (blank-vector 0 'float64) 0.0) X)))
   (row "non-finite coef"
        "coef: element 0: expected a finite value, got nan"
        (lambda () (go (linear-regression-result (vector '(+nan.0 2 3)) 0.0) X)))
   (row "non-finite intercept"
        "intercept: expected a finite value, got inf"
        (lambda () (go (linear-regression-result (vector '(1 2 3)) +inf.0) X)))
   (row "non-real intercept"
        "intercept: expected a real number, got 'b"
        (lambda () (go (linear-regression-result (vector '(1 2 3)) 'b) X)))
   (row "non-real intercept before the X views"
        "intercept: expected a real number, got 'b"
        (lambda () (go (linear-regression-result (vector '(1 2 3)) 'b) (blank-matrix 5 3 'int32))))
   (row "prediction overflow"
        "X: row 0: the prediction exceeds the float32 range"
        (lambda ()
          (go (linear-regression-result (vector '(1 1 1) 'float32) 0.0)
              (matrix '((3e38 3e38 3e38) (1 1 1)) 'float32))))))

(define (estimator-fit X y #:sample-weight [w #f] #:algorithm [a 'auto] #:fit-intercept [fi #t])
  (fit (linear-regression #:algorithm a #:fit-intercept fi) X y #:sample-weight w))

(test-gpu "linear-regression-fit refuses each row"
  (for ([r (in-list (fit-rows linear-regression-fit))])
    (test-case (row-id r)
      (check-row "linear-regression-fit" r))))

(test-gpu "the estimator's fit names the generic"
  (for ([r (in-list (fit-rows estimator-fit))])
    (test-case (row-id r)
      (check-row "fit" r))))

(test-gpu "linear-regression-predict refuses each row"
  (define fitted (linear-regression-fit (matrix grid) (vector targets)))
  (for ([r (in-list (append (predict-rows fitted linear-regression-predict)
                            (result-rows linear-regression-predict)))])
    (test-case (row-id r)
      (check-row "linear-regression-predict" r))))

(test-gpu "the estimator's predict names the generic"
  (define fitted (fit (linear-regression) (matrix grid) (vector targets)))
  (for ([r (in-list (predict-rows fitted predict))])
    (test-case (row-id r)
      (check-row "predict" r))))

(test-gpu "the estimator refuses a missing y and predict before fit"
  (check-cuml-error 'logic
                    "fit: linear-regression needs y"
                    (lambda () (fit (linear-regression) (matrix grid))))
  (check-cuml-error 'logic
                    "predict: linear-regression is not fitted"
                    (lambda () (predict (linear-regression) (matrix grid)))))

(define not-least-squares
  #rx"^linear-regression-fit: X: the [a-z-]+ solve is not a least-squares solution")

(define (collinear-rows n)
  (for/list ([i (in-range n)])
    (define a (- (modulo (* 37 i) 11) 5))
    (define b (- (modulo (* 53 i) 11) 5))
    (list a b (+ a b))))

(define (one-hot-rows n k)
  (for/list ([i (in-range n)])
    (for/list ([j (in-range k)])
      (if (= j (modulo i k)) 1 0))))

(define (noisy-targets rows)
  (for/list ([r (in-list rows)]
             [i (in-naturals)])
    (+ (for/sum ([x (in-list r)] [k (in-naturals 1)]) (* k x)) (/ (- (modulo (* 31 i) 17) 8) 16))))

(define (uneven-weights n)
  (define draw (uniform-source 77))
  (for/list ([i (in-range n)])
    (if (zero? (modulo i 13))
        0
        (dyadic (+ 1/2 (* 3/2 (draw))) 6))))

(test-gpu "a solve that is not a least-squares solution is refused"
  (for ([fixture (in-list (list (list 'eig 'float32 (collinear-rows 60) #f)
                                (list 'qr 'float32 (one-hot-rows 400 4) #f)
                                (list 'qr 'float64 (collinear-rows 60) #f)
                                (list 'auto 'float32 (collinear-rows 5003) (uneven-weights 5003))
                                (list 'qr 'float64 (collinear-rows 5003) (uneven-weights 5003))))])
    (match-define (list algorithm dt rows w) fixture)
    (test-case (format "~a ~a ~a rows~a" algorithm dt (length rows) (if w ", weighted" ""))
      (check-cuml-error 'logic
                        not-least-squares
                        (lambda ()
                          (linear-regression-fit (matrix rows dt)
                                                 (vector (noisy-targets rows) dt)
                                                 #:sample-weight (and w (vector w dt))
                                                 #:algorithm algorithm))))))

(test-gpu "X, y and w are unchanged after every fit"
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
