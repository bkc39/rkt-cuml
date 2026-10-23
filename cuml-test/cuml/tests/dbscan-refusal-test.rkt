#lang racket/base

(require (only-in compiler/find-exe find-exe)
         (only-in cuml/cluster dbscan-fit)
         (only-in racket/file delete-directory/files make-temporary-directory)
         (only-in racket/flonum fl-)
         (only-in racket/function curry)
         (only-in racket/list make-list)
         (only-in racket/match match match-define)
         (only-in racket/port port->string)
         (only-in rackunit check-equal? check-exn fail test-case)
         (only-in raft
                  device-resources
                  exn:fail:raft-kind
                  exn:fail:raft?
                  list*->device-matrix
                  with-device-resources)
         (only-in raft/private/resource with-release)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error raised)
         (only-in "private/dbscan-data.rkt" fitted-lists)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector dtypes))

(struct row (id message call))

(define out-dtypes '(int32 int64))

(define (other dt)
  (if (eq? dt 'float32) 'float64 'float32))

(define (grid dt)
  (list*->device-matrix (for/list ([i (in-range 10)])
                          (list (add1 i) (* 2 i)))
                        #:dtype dt))

(define (with-zero-row bad dt)
  (list*->device-matrix (list '(1 1) '(2 2) bad '(3 3)) #:dtype dt))

(define zero-row-message "X: row 2 is zero, so no cosine distance")

(define (fit-rows dt out)
  (define u (other dt))
  (define X (grid dt))
  (define fit (curry dbscan-fit #:out-dtype out))
  (match-define (list most past past-text)
    (match dt
      ['float32 (list "3.4028234663852886e+38" 1e300 "1e+300")]
      ['float64 (list "1.7976931348623157e+308" +inf.0 "inf")]))
  (list
   (row "D0" "X: expected float32 or float64, got int32" (lambda () (fit (blank-matrix 10 2 'int32))))
   (row "D1" "out-dtype: expected int32 or int64" (lambda () (dbscan-fit X #:out-dtype 'float32)))
   (row "D2" "metric: expected euclidean, cosine or precomputed" (lambda () (fit X #:metric 'l2)))
   (row "D3 rank" "X: expected rank 2, got rank 1" (lambda () (fit (blank-vector 10 dt))))
   (row "D3 layout"
        "X: expected row-major, got col-major 10x2"
        (lambda () (fit (blank-matrix 10 2 dt #:layout 'col-major))))
   (row "D4" "X: expected at least 1 row, got 0" (lambda () (fit (blank-matrix 0 2 dt))))
   (row "D5" "X: expected at least 1 column, got 0" (lambda () (fit (blank-matrix 10 0 dt))))
   (row "D6"
        "X: expected a square distance matrix, got 10x2"
        (lambda () (fit X #:metric 'precomputed)))
   (row "D8 dtype"
        (format "sample-weight: expected ~a, got ~a" dt u)
        (lambda () (fit X #:sample-weight (blank-vector 10 u))))
   (row "D8 length"
        "sample-weight: expected 10 elements, got 9"
        (lambda () (fit X #:sample-weight (blank-vector 9 dt))))
   (row "D8 rank"
        "sample-weight: expected rank 1, got rank 2"
        (lambda () (fit X #:sample-weight (blank-matrix 10 1 dt))))
   (row "D9" "eps: expected greater than 0, got 0" (lambda () (fit X #:eps 0)))
   (row "D9 negative" "eps: expected greater than 0, got -1" (lambda () (fit X #:eps -1)))
   (row "D9 -0" "eps: expected greater than 0, got 0" (lambda () (fit X #:eps -0.0)))
   (row "D9 nan" "eps: expected greater than 0, got nan" (lambda () (fit X #:eps +nan.0)))
   (row "D9 -nan"
        "eps: expected greater than 0, got nan"
        (lambda () (fit X #:eps (fl- +inf.0 +inf.0))))
   (row "D9 inf" (format "eps: expected at most ~a, got inf" most) (lambda () (fit X #:eps +inf.0)))
   (row "D9 past the dtype's largest"
        (format "eps: expected at most ~a, got ~a" most past-text)
        (lambda () (fit X #:eps past)))
   (row "D10" "min-samples: expected at least 1, got 0" (lambda () (fit X #:min-samples 0)))
   (row "D10 negative"
        "min-samples: expected at least 1, got -3"
        (lambda () (fit X #:min-samples -3)))
   (row "D11"
        "max-mbytes-per-batch: expected at least 0, got -1"
        (lambda () (fit X #:max-mbytes-per-batch -1)))
   (row "D12"
        "max-mbytes-per-batch: expected at most 18446744073709, got 18446744073710"
        (lambda () (fit X #:max-mbytes-per-batch 18446744073710)))
   (row "D14" zero-row-message (lambda () (fit (with-zero-row '(0 0) dt) #:metric 'cosine)))
   (row "D14 -0" zero-row-message (lambda () (fit (with-zero-row '(-0.0 -0.0) dt) #:metric 'cosine)))
   (row "D0 before D1"
        "X: expected float32 or float64, got int32"
        (lambda () (dbscan-fit (blank-matrix 10 2 'int32) #:out-dtype 'float32)))
   (row "D1 before D2"
        "out-dtype: expected int32 or int64"
        (lambda () (dbscan-fit X #:out-dtype 'float32 #:metric 'l2)))
   (row "D2 before D3"
        "metric: expected euclidean, cosine or precomputed"
        (lambda () (fit (blank-vector 10 dt) #:metric 'l2)))
   (row "D8 before D9"
        "sample-weight: expected 10 elements, got 9"
        (lambda () (fit X #:eps 0 #:sample-weight (blank-vector 9 dt))))
   (row "D9 before D10"
        "eps: expected greater than 0, got 0"
        (lambda () (fit X #:eps 0 #:min-samples 0)))
   (row "D10 before D11"
        "min-samples: expected at least 1, got 0"
        (lambda () (fit X #:min-samples 0 #:max-mbytes-per-batch -1)))
   (row "D12 before D14"
        "max-mbytes-per-batch: expected at most 18446744073709, got 18446744073710"
        (lambda ()
          (fit (with-zero-row '(0 0) dt) #:metric 'cosine #:max-mbytes-per-batch 18446744073710)))))

(test-gpu "dbscan-fit refuses each bad argument with its message, at both dtypes and both out-dtypes"
  (for* ([dt (in-list dtypes)]
         [out (in-list out-dtypes)]
         [r (in-list (fit-rows dt out))])
    (test-case (format "~a (~a, ~a)" (row-id r) dt out)
      (check-cuml-error 'logic (format "dbscan-fit: ~a" (row-message r)) (row-call r)))))

(define (with-extreme-row edge dt)
  (list*->device-matrix (append '((1 1) (1 1)) (list edge '(2 0)) (make-list 6 '(1 1))) #:dtype dt))

(define paired '(0 0 1 1 0 0 0 0 0 0))
(define joined '(0 0 0 -1 0 0 0 0 0 0))
(define noise '(0 0 -1 -1 0 0 0 0 0 0))

(define (extreme-rows dt)
  (append
   (list (list '(1e20 0) paired)
         (list '(1e19 0) paired)
         (list '(1e20 1e20) joined)
         (list '(+nan.0 1) noise)
         (list '(+inf.0 0) noise)
         (list '(-inf.0 +nan.0) noise))
   (match dt
     ['float32 (list (list '(1e-45 0) paired))]
     ['float64
      (list (list '(5e-324 0) paired) (list '(1e-160 1e-160) joined) (list '(1e200 0) paired))])))

(test-gpu "D14: a cosine row past float range, or with a NaN or an infinity, is not refused"
  (for* ([dt (in-list dtypes)]
         [c (in-list (extreme-rows dt))])
    (match-define (list edge labels) c)
    (test-case (format "~a ~a" dt edge)
      (match-define (list found _)
        (fitted-lists (with-extreme-row edge dt) #:eps 0.1 #:min-samples 2 #:metric 'cosine))
      (check-equal? found labels))))

(test-gpu "values that do not fit the native types raise exn:fail:contract"
  (define X (grid 'float64))
  (check-exn exn:fail:contract? (lambda () (dbscan-fit X #:min-samples 2.5)))
  (check-exn exn:fail:contract? (lambda () (dbscan-fit X #:min-samples (expt 2 31))))
  (check-exn exn:fail:contract? (lambda () (dbscan-fit X #:max-mbytes-per-batch (expt 2 63))))
  (check-exn exn:fail:contract? (lambda () (dbscan-fit X #:eps 'small))))

(test-gpu "released resources are refused under the procedure's own name"
  (define released (with-device-resources ([r (device-resources)]) r))
  (match (raised (lambda () (dbscan-fit (grid 'float64) #:resources released)))
    [(? exn:fail:raft? e)
     (check-equal? (exn:fail:raft-kind e) 'logic)
     (check-equal? (exn-message e) "dbscan-fit: the device resources on device 0 were released")]
    [other (fail (format "expected an exn:fail:raft, got ~e" other))]))

(define crash-program
  `(module crash racket/base
     (require (only-in cuml/cluster dbscan-fit)
              (only-in raft device-matrix exn:fail:raft-kind))
     (define (refusal dt out)
       (with-handlers ([exn:fail? (lambda (e) (list (exn:fail:raft-kind e) (exn-message e)))])
         (dbscan-fit (device-matrix 150000 2 #:dtype dt) #:max-mbytes-per-batch 1 #:out-dtype out)
         'accepted))
     (write (for*/list ([dt (in-list '(float32 float64))]
                        [out (in-list '(int32 int64))])
              (refusal dt out)))))

(define (run-crash-program)
  (with-release ([dir
                  (make-temporary-directory "cuml-dbscan-~a")
                  (lambda (dir) (delete-directory/files dir #:must-exist? #f))])
    (define path (build-path dir "crash.rkt"))
    (call-with-output-file path (lambda (out) (write crash-program out)))
    (define-values (process stdout stdin stderr) (subprocess #f #f #f (find-exe) path))
    (close-output-port stdin)
    (define output (port->string stdout))
    (define errors (port->string stderr))
    (subprocess-wait process)
    (list (subprocess-status process) output errors)))

(test-gpu "a batch too small for cuML is refused, where cuML would die of SIGFPE"
  (match (run-crash-program)
    [(list 0 output _)
     (check-equal?
      (read (open-input-string output))
      (for*/list ([_ (in-list dtypes)]
                  [least (in-list '(2 3))])
        (list 'logic
              (format "dbscan-fit: max-mbytes-per-batch: expected at least ~a for 150000 rows, got 1"
                      least))))]
    [(list status _ errors) (fail (format "the fit's process exited with ~a:\n~a" status errors))]))
