#lang racket/base

(require (only-in cuml/cluster kmeans-fit kmeans-predict kmeans-result? kmeans-score kmeans-transform)
         (only-in racket/flonum fl-)
         (only-in racket/match match)
         (only-in rackunit check-equal? check-pred fail test-case)
         (only-in raft
                  device-resources
                  exn:fail:raft-kind
                  exn:fail:raft?
                  list*->device-matrix
                  with-device-resources)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error raised)
         (only-in "private/fitted.rkt" fitted)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector dtypes))

(struct row (id message call))

(define (other dt)
  (if (eq? dt 'float32) 'float64 'float32))

(define (grid dt)
  (list*->device-matrix (for/list ([i (in-range 10)])
                          (list i (* 2 i)))
                        #:dtype dt))

(define (fit-rows dt)
  (define u (other dt))
  (define X (grid dt))
  (define fit (fitted kmeans-fit values))
  (define fit3 (fitted kmeans-fit values #:n-clusters 3))
  (list
   (row "F1" "X: expected float32 or float64, got int32" (fit3 (blank-matrix 10 2 'int32)))
   (row "F2" "X: expected rank 2, got rank 1" (fit3 (blank-vector 10 dt)))
   (row "F3"
        "X: expected row-major, got col-major 10x2"
        (fit3 (blank-matrix 10 2 dt #:layout 'col-major)))
   (row "F5" "X: expected at least 1 column, got 0" (fit3 (blank-matrix 10 0 dt)))
   (row "F6" "n-clusters: expected at least 1, got 0" (fit X #:n-clusters 0))
   (row "F7" "X: 10 samples are fewer than the 11 clusters" (fit X #:n-clusters 11))
   (row "F8"
        (format "sample-weight: expected ~a, got ~a" dt u)
        (fit3 X #:sample-weight (blank-vector 10 u)))
   (row "F9"
        "sample-weight: expected 10 elements, got 9"
        (fit3 X #:sample-weight (blank-vector 9 dt)))
   (row "F10"
        "init: expected scalable-k-means++, k-means++, random or a centroids matrix"
        (fit3 X #:init (string->symbol "k-means||")))
   (row "F12" (format "init: expected ~a, got ~a" dt u) (fit X #:init (blank-matrix 3 2 u)))
   (row "F13" "init: expected 3 rows, got 2" (fit3 X #:init (blank-matrix 2 2 dt)))
   (row "F14" "init: expected 2 columns, got 3" (fit X #:init (blank-matrix 3 3 dt)))
   (row "F15"
        "init: expected row-major, got col-major 3x2"
        (fit X #:init (blank-matrix 3 2 dt #:layout 'col-major)))
   (row "F16" "max-iter: expected at least 0, got -1" (fit3 X #:max-iter -1))
   (row "F17" "n-init: expected at least 1, got 0" (fit3 X #:n-init 0))
   (row "F18"
        "max-samples-per-batch: expected at least 0, got -1"
        (fit3 X #:max-samples-per-batch -1))
   (row "F21" "tol: expected greater than 0, got 0" (fit3 X #:tol 0))
   (row "F21 nan" "tol: expected greater than 0, got nan" (fit3 X #:tol +nan.0))
   (row "F21 -nan" "tol: expected greater than 0, got nan" (fit3 X #:tol (fl- +inf.0 +inf.0)))
   (row "F21 -0" "tol: expected greater than 0, got 0" (fit3 X #:tol -0.0))
   (row "F22" "oversampling-factor: expected at least 0, got -1" (fit3 X #:oversampling-factor -1))
   (row "F22 nan"
        "oversampling-factor: expected at least 0, got nan"
        (fit3 X #:oversampling-factor +nan.0))
   (row "F1 before F6"
        "X: expected float32 or float64, got int32"
        (fit (blank-matrix 10 2 'int32) #:n-clusters 0))))

(define ((on centroids X) proc)
  (proc centroids X))

(define (apply-rows dt)
  (define u (other dt))
  (define X (grid dt))
  (define C (blank-matrix 3 2 dt))
  (list
   (row "P1" "centroids: expected float32 or float64, got int32" (on (blank-matrix 3 2 'int32) X))
   (row "P2 rank" "centroids: expected rank 2, got rank 1" (on (blank-vector 3 dt) X))
   (row "P2 layout"
        "centroids: expected row-major, got col-major 3x2"
        (on (blank-matrix 3 2 dt #:layout 'col-major) X))
   (row "P3 rows" "centroids: expected at least 1 row, got 0" (on (blank-matrix 0 2 dt) X))
   (row "P3 columns" "centroids: expected at least 1 column, got 0" (on (blank-matrix 3 0 dt) X))
   (row "P4" (format "X: expected ~a, got ~a" dt u) (on C (blank-matrix 10 2 u)))
   (row "P5 columns" "X: expected 2 columns, got 3" (on C (blank-matrix 10 3 dt)))
   (row "P5 layout"
        "X: expected row-major, got col-major 10x2"
        (on C (blank-matrix 10 2 dt #:layout 'col-major)))
   (row "P5 rank" "X: expected rank 2, got rank 1" (on C (blank-vector 10 dt)))
   (row "P6" "X: expected at least 1 row, got 0" (on C (blank-matrix 0 2 dt)))
   (row "F18"
        "max-samples-per-batch: expected at least 0, got -1"
        (lambda (proc) (proc C X #:max-samples-per-batch -1)))
   (row "P1 before P4"
        "centroids: expected float32 or float64, got int32"
        (on (blank-matrix 3 2 'int32) (blank-matrix 10 2 u)))))

(define (score-rows dt)
  (define u (other dt))
  (define X (grid dt))
  (define C (blank-matrix 3 2 dt))
  (list (row "P7 dtype"
             (format "sample-weight: expected ~a, got ~a" dt u)
             (lambda () (kmeans-score C X #:sample-weight (blank-vector 10 u))))
        (row "P7 length"
             "sample-weight: expected 10 elements, got 9"
             (lambda () (kmeans-score C X #:sample-weight (blank-vector 9 dt))))))

(define (check-row who r thunk)
  (check-cuml-error 'logic (format "~a: ~a" who (row-message r)) thunk))

(define appliers
  (hasheq 'kmeans-predict kmeans-predict
          'kmeans-transform kmeans-transform
          'kmeans-score kmeans-score))

(test-gpu "kmeans-fit refuses each bad argument with its message, float32 and float64"
  (for* ([dt (in-list dtypes)]
         [r (in-list (fit-rows dt))])
    (test-case (format "~a (~a)" (row-id r) dt)
      (check-row 'kmeans-fit r (row-call r)))))

(test-gpu "kmeans-predict, kmeans-transform and kmeans-score refuse each bad argument alike"
  (for* ([dt (in-list dtypes)]
         [r (in-list (apply-rows dt))]
         [(who proc) (in-hash appliers)])
    (test-case (format "~a ~a (~a)" who (row-id r) dt)
      (check-row who r (lambda () ((row-call r) proc))))))

(test-gpu "k-means++ ignores the oversampling factor"
  (for ([dt (in-list dtypes)])
    (check-pred kmeans-result?
                (kmeans-fit (grid dt) #:n-clusters 3 #:init 'k-means++ #:oversampling-factor -1))))

(test-gpu "kmeans-transform refuses distances past the index type"
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (check-cuml-error 'logic
                        "kmeans-transform: distances: 100000x50000 elements do not fit the index type"
                        (lambda ()
                          (kmeans-transform (blank-matrix 50000 4 dt) (blank-matrix 100000 4 dt)))))))

(test-gpu "released resources are refused under the procedure's own name"
  (define X (grid 'float64))
  (define C (blank-matrix 3 2 'float64))
  (define released (with-device-resources ([r (device-resources)]) r))
  (define calls
    (hasheq 'kmeans-fit (lambda () (kmeans-fit X #:n-clusters 3 #:resources released))
            'kmeans-predict (lambda () (kmeans-predict C X #:resources released))
            'kmeans-transform (lambda () (kmeans-transform C X #:resources released))
            'kmeans-score (lambda () (kmeans-score C X #:resources released))))
  (for ([(who thunk) (in-hash calls)])
    (test-case (symbol->string who)
      (match (raised thunk)
        [(? exn:fail:raft? e)
         (check-equal? (exn:fail:raft-kind e) 'logic)
         (check-equal? (exn-message e)
                       (format "~a: the device resources on device 0 were released" who))]
        [other (fail (format "expected an exn:fail:raft, got ~e" other))]))))

(test-gpu "kmeans-score refuses bad sample weights"
  (for* ([dt (in-list dtypes)]
         [r (in-list (score-rows dt))])
    (test-case (format "~a (~a)" (row-id r) dt)
      (check-row 'kmeans-score r (row-call r)))))
