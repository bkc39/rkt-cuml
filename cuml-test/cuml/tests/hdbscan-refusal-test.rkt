#lang racket/base

(require (only-in cuml native-model-release!)
         (only-in cuml/cluster
                  hdbscan-all-points-membership-vectors
                  hdbscan-approximate-predict
                  hdbscan-fit
                  hdbscan-membership-vector
                  hdbscan-params
                  hdbscan-result-model)
         (only-in racket/function curry)
         (only-in racket/match match)
         (only-in rackunit check-equal? check-exn fail test-case)
         (only-in raft
                  device-resources
                  exn:fail:raft-kind
                  exn:fail:raft?
                  list*->device-matrix
                  with-device-resources)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error raised)
         (only-in "private/dbscan-data.rkt" mat)
         (only-in "private/hdbscan-data.rkt" dataset)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "private/program.rkt" run-program)
         (only-in "private/refusal-rows.rkt" check-row norm-message row row-id with-row))

(define (grid dt [n 10])
  (list*->device-matrix (for/list ([i (in-range n)])
                          (list (add1 i) (* 2 i)))
                        #:dtype dt))

(define (with-bad-row bad)
  (with-row bad #:then '((7 7) (8 9) (9 1) (3 3))))

(define flt-most "expected at most 3.4028234663852886e+38, got ")

(define (fit-rows)
  (define X (grid 'float32))
  (define fit hdbscan-fit)
  (list
   (row "H0 float64" "X: expected float32, got float64" (lambda () (fit (grid 'float64))))
   (row "H0 int32" "X: expected float32, got int32" (lambda () (fit (blank-matrix 10 2 'int32))))
   (row "H1 rank" "X: expected rank 2, got rank 1" (lambda () (fit (blank-vector 10 'float32))))
   (row "H1 layout"
        "X: expected row-major, got col-major 10x2"
        (lambda () (fit (blank-matrix 10 2 'float32 #:layout 'col-major))))
   (row "H2" "X: expected at least 2 rows, got 1" (lambda () (fit (grid 'float32 1))))
   (row "H2 empty" "X: expected at least 2 rows, got 0" (lambda () (fit (blank-matrix 0 2 'float32))))
   (row "H3" "X: expected at least 1 column, got 0" (lambda () (fit (blank-matrix 10 0 'float32))))
   (row "H7 zero" "min-samples: expected at least 1, got 0" (lambda () (fit X #:min-samples 0)))
   (row "H7 most" "min-samples: expected at most 1023, got 1024" (lambda () (fit X #:min-samples 1024)))
   (row "H8" "X: 4 samples are fewer than min-samples 5" (lambda () (fit (grid 'float32 4))))
   (row "H8 explicit"
        "X: 10 samples are fewer than min-samples 11"
        (lambda () (fit X #:min-samples 11)))
   (row "H9 negative"
        "cluster-selection-epsilon: expected at least 0, got -1"
        (lambda () (fit X #:cluster-selection-epsilon -1)))
   (row "H9 nan"
        "cluster-selection-epsilon: expected at least 0, got nan"
        (lambda () (fit X #:cluster-selection-epsilon +nan.0)))
   (row "H9 inf"
        (string-append "cluster-selection-epsilon: " flt-most "inf")
        (lambda () (fit X #:cluster-selection-epsilon +inf.0)))
   (row "H10" "max-cluster-size: expected at least 0, got -1" (lambda () (fit X #:max-cluster-size -1)))
   (row "H11 zero" "alpha: expected greater than 0, got 0" (lambda () (fit X #:alpha 0)))
   (row "H11 negative" "alpha: expected greater than 0, got -1" (lambda () (fit X #:alpha -1)))
   (row "H11 nan" "alpha: expected greater than 0, got nan" (lambda () (fit X #:alpha +nan.0)))
   (row "H11 inf" (string-append "alpha: " flt-most "inf") (lambda () (fit X #:alpha +inf.0)))
   (row "H12"
        "cluster-selection-method: expected eom or leaf"
        (lambda () (fit X #:cluster-selection-method 'average)))
   (row "H13 nan" (norm-message "nan") (lambda () (fit (with-bad-row '(+nan.0 1)))))
   (row "H13 inf" (norm-message "inf") (lambda () (fit (with-bad-row '(+inf.0 0)))))
   (row "H13 -inf" (norm-message "inf") (lambda () (fit (with-bad-row '(0 -inf.0)))))
   (row "H13 1e20" (norm-message "1.0000000400817551e+40") (lambda () (fit (with-bad-row '(1e20 0)))))
   (row "H13 2^63"
        (norm-message "8.507059173023462e+37")
        (lambda () (fit (with-bad-row (list (expt 2 63) 0)))))
   (row "H0 before H2" "X: expected float32, got float64" (lambda () (fit (grid 'float64 1))))
   (row "H8 before H13"
        "X: 10 samples are fewer than min-samples 11"
        (lambda () (fit (with-bad-row '(+nan.0 1)) #:min-samples 11)))
   (row "H9 before H11"
        "cluster-selection-epsilon: expected at least 0, got -1"
        (lambda () (fit X #:cluster-selection-epsilon -1 #:alpha 0)))))

(define (prediction-rows result)
  (define predict (curry hdbscan-approximate-predict result))
  (define membership (curry hdbscan-membership-vector result))
  (define (both id message points)
    (list (row (format "~a hdbscan-approximate-predict" id) message (lambda () (predict points)))
          (row (format "~a hdbscan-membership-vector" id) message (lambda () (membership points)))))
  (append
   (both "A1" "points: expected float32, got float64" (grid 'float64 4))
   (both "A2 rank" "points: expected rank 2, got rank 1" (blank-vector 4 'float32))
   (both "A2 layout"
         "points: expected row-major, got col-major 4x2"
         (blank-matrix 4 2 'float32 #:layout 'col-major))
   (both "A3" "points: expected at least 1 row, got 0" (blank-matrix 0 2 'float32))
   (both "A4"
         "points: expected 2 columns, as the model was fit on, got 3"
         (blank-matrix 4 3 'float32))
   (both "A4 before A8"
         "points: expected 2 columns, as the model was fit on, got 3"
         (mat '((1 2 3) (+nan.0 0 0)) 'float32))
   (list (row "M1 hdbscan-membership-vector"
              "batch-size: expected at least 1, got 0"
              (lambda () (membership (grid 'float32 4) #:batch-size 0)))
         (row "M1 hdbscan-all-points-membership-vectors"
              "batch-size: expected at least 1, got 0"
              (lambda () (hdbscan-all-points-membership-vectors result #:batch-size 0))))))

(test-gpu "R1: hdbscan-fit refuses each bad argument with its message"
  (for ([r (in-list (fit-rows))])
    (test-case (row-id r)
      (check-row 'hdbscan-fit r))))

(test-gpu "R1: the prediction procedures refuse each bad argument with their own names"
  (define result (hdbscan-fit (mat (dataset 'b50) 'float32)))
  (for ([r (in-list (prediction-rows result))])
    (test-case (row-id r)
      (match (regexp-match #rx" ([a-z-]+)$" (row-id r))
        [(list _ name) (check-row (string->symbol name) r)]))))

(test-gpu "A0: a released model is refused by every prediction procedure before its points are read"
  (define result (hdbscan-fit (mat (dataset 'b50) 'float32)))
  (native-model-release! (hdbscan-result-model result))
  (define float64 (grid 'float64 4))
  (for ([name (in-list '(hdbscan-approximate-predict hdbscan-membership-vector
                                                     hdbscan-all-points-membership-vectors))]
        [call (in-list (list (lambda () (hdbscan-approximate-predict result float64))
                             (lambda () (hdbscan-membership-vector result float64))
                             (lambda () (hdbscan-all-points-membership-vectors result))))])
    (check-cuml-error 'logic
                      (format "~a: model: expected a live model, got a released one" name)
                      call)))

(define crash-program
  `(module crash racket/base
     (require (only-in cuml/cluster
                       hdbscan-approximate-predict
                       hdbscan-fit
                       hdbscan-membership-vector)
              (only-in raft exn:fail:raft-kind list*->device-matrix))
     (define (outcome thunk)
       (with-handlers ([exn:fail? (lambda (e) (list (exn:fail:raft-kind e) (exn-message e)))])
         (thunk)
         'accepted))
     (define (rows n)
       (list*->device-matrix (for/list ([i (in-range n)])
                               (list (* 3 i) (* i i)))
                             #:dtype 'float32))
     (define model (hdbscan-fit (rows 30)))
     (define short (hdbscan-fit (rows 9) #:min-samples 5 #:min-cluster-size 2))
     (define enough (hdbscan-fit (rows 10) #:min-samples 5 #:min-cluster-size 2))
     (define (bad v)
       (list*->device-matrix (list '(1 1) '(2 3) (list v 0) '(4 4)) #:dtype 'float32))
     (write
      (append
       (for/list ([mcs (in-list '(1 0 -3))])
         (outcome (lambda () (hdbscan-fit (rows 30) #:min-cluster-size mcs #:min-samples 5))))
       (for*/list ([v (in-list (list +nan.0 +inf.0 -inf.0 1e20))]
                   [predict (in-list (list hdbscan-approximate-predict hdbscan-membership-vector))])
         (outcome (lambda () (predict model (bad v)))))
       (for/list ([predict (in-list (list hdbscan-approximate-predict hdbscan-membership-vector))])
         (outcome (lambda () (predict short (rows 3)))))
       (list (outcome (lambda () (hdbscan-approximate-predict enough (rows 3)))))))))

(define (norm-of got)
  (format "points: row 2: expected a squared norm of at most 4.253529586511731e+37, got ~a" got))

(test-gpu "R2, R6: crash-class refusals, in a process of their own"
  (match (run-program crash-program)
    [(list 0 output _)
     (check-equal?
      (read (open-input-string output))
      (append
       (for/list ([mcs (in-list '(1 0 -3))])
         (list 'logic (format "hdbscan-fit: min-cluster-size: expected at least 2, got ~a" mcs)))
       (for*/list ([got (in-list '("nan" "inf" "inf" "1.0000000400817551e+40"))]
                   [name (in-list '(hdbscan-approximate-predict hdbscan-membership-vector))])
         (list 'logic (format "~a: ~a" name (norm-of got))))
       (for/list ([name (in-list '(hdbscan-approximate-predict hdbscan-membership-vector))])
         (list 'logic
               (format "~a: model: prediction needs at least 10 training rows (2 × min-samples), got 9"
                       name)))
       (list 'accepted)))]
    [(list status _ errors) (fail (format "the refusals' process exited with ~a:\n~a" status errors))]))

(test-gpu "R2: rows at the squared-norm boundary, and subnormal rows, fit and predict"
  (define boundary
    (list*->device-matrix
     (append
      (list '(0 0) (list (expt 2 62) (expt 2 62)) (list (- (expt 2 62)) (- (expt 2 62))) '(1e-45 0))
      (for/list ([i (in-range 6)])
        (list i (* i i))))
     #:dtype 'float32))
  (define result (hdbscan-fit boundary #:min-cluster-size 2 #:min-samples 2))
  (void (hdbscan-approximate-predict result boundary)))

(test-case "R4: a fixed key other than its value is refused before X is read"
  (for ([params (in-list (list (hdbscan-params #:metric 'cosine)
                               (hdbscan-params #:build-algo 'nn-descent)))]
        [message (in-list '("metric: expected euclidean, got cosine"
                            "build-algo: expected brute-force, got nn-descent"))])
    (check-cuml-error 'logic
                      (string-append "hdbscan-fit: " message)
                      (lambda () (hdbscan-fit 'X #:params params #:resources 'unused)))))

(test-gpu "R4: values that do not fit the native types raise exn:fail:contract"
  (define X (grid 'float32))
  (define fit (curry hdbscan-fit X))
  (check-exn exn:fail:contract? (lambda () (fit #:max-cluster-size (expt 2 63))))
  (check-exn exn:fail:contract? (lambda () (fit #:min-cluster-size 2.5)))
  (check-exn exn:fail:contract? (lambda () (fit #:alpha 'one)))
  (define result (hdbscan-fit (mat (dataset 'b50) 'float32)))
  (check-exn exn:fail:contract?
             (lambda () (hdbscan-membership-vector result (grid 'float32) #:batch-size (expt 2 63))))
  (check-exn exn:fail:contract?
             (lambda () (hdbscan-all-points-membership-vectors result #:batch-size (expt 2 63)))))

(test-gpu "released resources are refused under the procedure's own name"
  (define released (with-device-resources ([r (device-resources)]) r))
  (match (raised (lambda () (hdbscan-fit (grid 'float32) #:resources released)))
    [(? exn:fail:raft? e)
     (check-equal? (exn:fail:raft-kind e) 'logic)
     (check-equal? (exn-message e) "hdbscan-fit: the device resources on device 0 were released")]
    [other (fail (format "expected an exn:fail:raft, got ~e" other))]))
