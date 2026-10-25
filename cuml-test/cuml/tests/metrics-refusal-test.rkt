#lang racket/base

(require (only-in cuml/metrics
                  adjusted-rand-score
                  contingency-matrix
                  silhouette-samples
                  silhouette-score
                  trustworthiness)
         (only-in racket/list make-list range)
         (only-in racket/match match)
         (only-in rackunit check-equal? check-exn test-case)
         (only-in raft list*->device-matrix)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "private/metrics-data.rkt" labels-of)
         (only-in "private/refusal-rows.rkt" check-row row row-id))

(define float-most "4.253529586511731e+37")
(define double-most "2.247116418577895e+307")

(define (norm-message name i most got)
  (format "~a: row ~a: expected a squared norm of at most ~a, got ~a" name i most got))

(define (rows-with n bad #:at [at 4] #:dtype [dt 'float32])
  (list*->device-matrix (for/list ([i (in-range n)])
                          (if (= i at)
                              bad
                              (list (add1 (modulo i 7)) (* 3 i))))
                        #:dtype dt))

(define (distinct-rows n)
  (list*->device-matrix (for/list ([i (in-range n)])
                          (list (* i i) (add1 (* 3 i))))
                        #:dtype 'float32))

(define (label-rows proc)
  (define a (labels-of '(0 0 1 1 2) 'int32))
  (define b (labels-of '(3 3 4 4 4) 'int64))
  (list
   (row "L1 true"
        "labels-true: expected int32 or int64, got float32"
        (lambda () (proc (blank-vector 5 'float32) b)))
   (row "L1 pred"
        "labels-pred: expected int32 or int64, got float64"
        (lambda () (proc a (blank-vector 5 'float64))))
   (row "L2"
        "labels-true: expected rank 1, got rank 2"
        (lambda () (proc (blank-matrix 5 1 'int32) b)))
   (row "L4"
        "labels-pred: expected 5 elements, got 4"
        (lambda () (proc a (labels-of '(1 2 3 4) 'int32))))
   (row "C1 true"
        "labels-true: expected at most 46340 distinct labels, got 46341"
        (lambda () (proc (labels-of (range 46341) 'int32) (labels-of (make-list 46341 0) 'int32))))
   (row "C1 pred"
        "labels-pred: expected at most 46340 distinct labels, got 46341"
        (lambda () (proc (labels-of (make-list 46341 0) 'int64) (labels-of (range 46341) 'int64))))
   (row "L1 before L4"
        "labels-true: expected int32 or int64, got float32"
        (lambda () (proc (blank-vector 5 'float32) (labels-of '(1) 'int32))))))

(test-gpu "R1: contingency-matrix and adjusted-rand-score refuse each bad label argument"
  (for* ([proc (in-list (list contingency-matrix adjusted-rand-score))]
         [r (in-list (label-rows proc))])
    (test-case (format "~a ~a" (object-name proc) (row-id r))
      (check-row (object-name proc) r))))

(define metric-message "metric: expected euclidean, sqeuclidean, manhattan, cosine or chebyshev")

(define (silhouette-rows proc)
  (define X (rows-with 30 '(1 1)))
  (define two (labels-of (append '(0 1) (make-list 28 1)) 'int32))
  (define (cosine-rows bad #:dtype [dt 'float32])
    (list*->device-matrix (for/list ([i (in-range 30)])
                            (match (assv i bad)
                              [(list _ row) row]
                              [#f (list (add1 i) 1)]))
                          #:dtype dt))
  (define zero-message "X: row 3 is zero, so no cosine distance")
  (list
   (row "S11" metric-message (lambda () (proc X two #:metric 'l2)))
   (row "S9" "chunk-size: expected at least 1, got 0" (lambda () (proc X two #:chunk-size 0)))
   (row "S9 negative"
        "chunk-size: expected at least 1, got -1"
        (lambda () (proc X two #:chunk-size -1)))
   (row "S0"
        "X: expected float32 or float64, got int32"
        (lambda () (proc (blank-matrix 30 2 'int32) two)))
   (row "S1 rank" "X: expected rank 2, got rank 1" (lambda () (proc (blank-vector 30 'float32) two)))
   (row "S1 layout"
        "X: expected row-major, got col-major 30x2"
        (lambda () (proc (blank-matrix 30 2 'float32 #:layout 'col-major) two)))
   (row "L1"
        "labels: expected int32 or int64, got float32"
        (lambda () (proc X (blank-vector 30 'float32))))
   (row "L4"
        "labels: expected 30 elements, got 29"
        (lambda () (proc X (labels-of (make-list 29 0) 'int32))))
   (row "S3"
        "X: expected at least 1 column, got 0"
        (lambda () (proc (blank-matrix 30 0 'float32) two)))
   (row "S5"
        "X: expected at least 3 rows, got 2"
        (lambda () (proc (rows-with 2 '(1 1)) (labels-of '(0 1) 'int32))))
   (row "S6"
        "labels: expected at least 2 distinct labels, got 1"
        (lambda () (proc X (labels-of (make-list 30 7) 'int64))))
   (row "S7"
        "labels: expected at most 29 distinct labels, got 30"
        (lambda () (proc X (labels-of (range 30) 'int32))))
   (row "S12 nan"
        (norm-message "X" 4 float-most "nan")
        (lambda () (proc (rows-with 30 '(+nan.0 1)) two)))
   (row "S12 float64 inf"
        (norm-message "X" 4 double-most "inf")
        (lambda () (proc (rows-with 30 '(+inf.0 1) #:dtype 'float64) two)))
   (row "S13" zero-message (lambda () (proc (cosine-rows '((3 (0 0)))) two #:metric 'cosine)))
   (row "S13 float32 underflow"
        zero-message
        (lambda () (proc (cosine-rows '((3 (1e-30 0)))) two #:metric 'cosine)))
   (row "S13 float64 underflow"
        zero-message
        (lambda () (proc (cosine-rows '((3 (1e-170 0))) #:dtype 'float64) two #:metric 'cosine)))
   (row "S11 before S9" metric-message (lambda () (proc X two #:metric 'l1 #:chunk-size 0)))
   (row "S9 before S0"
        "chunk-size: expected at least 1, got 0"
        (lambda () (proc (blank-matrix 30 2 'int32) two #:chunk-size 0)))
   (row "L4 before S5"
        "labels: expected 2 elements, got 1"
        (lambda () (proc (rows-with 2 '(1 1)) (labels-of '(0) 'int32))))
   (row "S5 before S6"
        "X: expected at least 3 rows, got 2"
        (lambda () (proc (rows-with 2 '(1 1)) (labels-of '(0 0) 'int32))))
   (row "S6 before S12"
        "labels: expected at least 2 distinct labels, got 1"
        (lambda () (proc (rows-with 30 '(+nan.0 1)) (labels-of (make-list 30 0) 'int32))))
   (row "S12 before S13"
        (norm-message "X" 3 float-most "nan")
        (lambda () (proc (cosine-rows '((2 (0 0)) (3 (+nan.0 0)))) two #:metric 'cosine)))))

(test-gpu "R1: silhouette-score and silhouette-samples refuse each bad argument"
  (for* ([proc (in-list (list silhouette-score silhouette-samples))]
         [r (in-list (silhouette-rows proc))])
    (test-case (format "~a ~a" (object-name proc) (row-id r))
      (check-row (object-name proc) r))))

(define (trust-rows)
  (define X (distinct-rows 100))
  (define E (distinct-rows 100))
  (list (row "T0" "metric: expected euclidean" (lambda () (trustworthiness X E #:metric 'cosine)))
        (row "T1"
             "batch-size: expected at least 1, got 0"
             (lambda () (trustworthiness X E #:batch-size 0)))
        (row "T6"
             "n-neighbors: expected at least 1, got 0"
             (lambda () (trustworthiness X E #:n-neighbors 0)))
        (row "T2 X"
             "X: expected float32, got float64"
             (lambda () (trustworthiness (blank-matrix 100 2 'float64) E)))
        (row "T2 X-embedded"
             "X-embedded: expected float32, got float64"
             (lambda () (trustworthiness X (blank-matrix 100 2 'float64))))
        (row "T3 X"
             "X: expected rank 2, got rank 1"
             (lambda () (trustworthiness (blank-vector 100 'float32) E)))
        (row "T3 X-embedded"
             "X-embedded: expected row-major, got col-major 100x2"
             (lambda () (trustworthiness X (blank-matrix 100 2 'float32 #:layout 'col-major))))
        (row "T4"
             "X-embedded: expected 100 rows, got 99"
             (lambda () (trustworthiness X (distinct-rows 99))))
        (row "T5 X"
             "X: expected at least 1 column, got 0"
             (lambda () (trustworthiness (blank-matrix 100 0 'float32) E)))
        (row "T5 X-embedded"
             "X-embedded: expected at least 1 column, got 0"
             (lambda () (trustworthiness X (blank-matrix 100 0 'float32))))
        (row "T7"
             "n-neighbors: expected less than half the 20 rows, got 10"
             (lambda () (trustworthiness (distinct-rows 20) (distinct-rows 20) #:n-neighbors 10)))
        (row "T9 X"
             (norm-message "X" 4 float-most "nan")
             (lambda () (trustworthiness (rows-with 100 '(+nan.0 0)) E)))
        (row "T9 X-embedded"
             (norm-message "X-embedded" 4 float-most "1.0000000400817551e+40")
             (lambda () (trustworthiness X (rows-with 100 '(1e20 0)))))
        (row "T0 before T1"
             "metric: expected euclidean"
             (lambda () (trustworthiness X E #:metric 'l2 #:batch-size 0)))
        (row "T1 before T6"
             "batch-size: expected at least 1, got 0"
             (lambda () (trustworthiness X E #:batch-size 0 #:n-neighbors 0)))
        (row "T6 before T2"
             "n-neighbors: expected at least 1, got 0"
             (lambda () (trustworthiness (blank-matrix 100 2 'float64) E #:n-neighbors 0)))
        (row "T4 before T7"
             "X-embedded: expected 20 rows, got 100"
             (lambda () (trustworthiness (distinct-rows 20) E #:n-neighbors 10)))
        (row "T7 before T9"
             "n-neighbors: expected less than half the 100 rows, got 50"
             (lambda () (trustworthiness (rows-with 100 '(+nan.0 0)) E #:n-neighbors 50)))))

(test-gpu "R1: trustworthiness refuses each bad argument"
  (for ([r (in-list (trust-rows))])
    (test-case (row-id r)
      (check-row 'trustworthiness r))))

(test-gpu "R2: rows at the squared-norm bound are scored, rows past it refused"
  (define two (labels-of (append '(0 1) (make-list 28 1)) 'int32))
  (for ([proc (in-list (list silhouette-score silhouette-samples))])
    (define (refuses message thunk)
      (check-cuml-error 'logic (format "~a: ~a" (object-name proc) message) thunk))
    (proc (rows-with 30 (list (expt 2 62) (expt 2 62))) two)
    (proc (rows-with 30 (list (expt 2 510) (expt 2 510)) #:dtype 'float64) two)
    (refuses (norm-message "X" 4 float-most "8.507059173023462e+37")
             (lambda () (proc (rows-with 30 (list (expt 2 63) 0)) two)))
    (refuses (norm-message "X" 4 double-most "4.49423283715579e+307")
             (lambda () (proc (rows-with 30 (list (expt 2 511) 0) #:dtype 'float64) two)))
    (refuses (norm-message "X" 4 float-most "1.0000000400817551e+40")
             (lambda () (proc (rows-with 30 '(1e20 0)) two))))
  (define X (distinct-rows 100))
  (trustworthiness (rows-with 100 (list (expt 2 62) (expt 2 62))) X)
  (for ([bad (in-list '((+nan.0 0) (0 +inf.0) (1e20 0)))])
    (check-cuml-error 'logic
                      #rx"^trustworthiness: X: row 4: "
                      (lambda () (trustworthiness (rows-with 100 bad) X)))
    (check-cuml-error 'logic
                      #rx"^trustworthiness: X-embedded: row 4: "
                      (lambda () (trustworthiness X (rows-with 100 bad))))
    (check-equal? (trustworthiness X X) 1.0 "a valid call after the refusal runs")))

(test-gpu "R3: chunk-size 0 and -1 and batch-size 0 refuse at once"
  (define X (rows-with 30 '(1 1)))
  (define two (labels-of (append '(0 1) (make-list 28 1)) 'int32))
  (for ([chunk (in-list '(0 -1))])
    (check-cuml-error 'logic
                      (format "silhouette-score: chunk-size: expected at least 1, got ~a" chunk)
                      (lambda () (silhouette-score X two #:chunk-size chunk))))
  (check-cuml-error 'logic
                    "trustworthiness: batch-size: expected at least 1, got 0"
                    (lambda ()
                      (trustworthiness (distinct-rows 30) (distinct-rows 30) #:batch-size 0))))

(test-gpu "R4: the index bounds are refused on the host"
  (check-cuml-error
   'logic
   "adjusted-rand-score: labels-true: expected at most 46340 distinct labels, got 46341"
   (lambda ()
     (adjusted-rand-score (labels-of (range 46341) 'int32) (labels-of (make-list 46341 0) 'int32))))
  (check-cuml-error
   'logic
   "silhouette-score: labels: 46342 rows with 46341 distinct labels need 2147534622 entries, more than 2147483647"
   (lambda ()
     (silhouette-score (list*->device-matrix (make-list 46342 '(1)) #:dtype 'float32)
                       (labels-of (append (range 46341) '(0)) 'int32))))
  (check-cuml-error 'logic
                    "silhouette-score: chunk-size: expected at most 46340, got 46341"
                    (lambda ()
                      (silhouette-score (blank-matrix 46341 1 'float32)
                                        (blank-vector 46341 'int32)
                                        #:chunk-size 46341)))
  (check-cuml-error
   'logic
   "trustworthiness: n-neighbors: 65536 rows with 32768 neighbours each need 2147483648 entries, more than 2147483647"
   (lambda ()
     (trustworthiness (blank-matrix 65536 1 'float32)
                      (blank-matrix 65536 1 'float32)
                      #:n-neighbors 32767)))
  (check-cuml-error
   'logic
   "trustworthiness: batch-size: 46341 rows of 46341 distances need 2147488281 entries, more than 2147483647"
   (lambda ()
     (trustworthiness (blank-matrix 46341 1 'float32)
                      (blank-matrix 46341 1 'float32)
                      #:batch-size 46341))))

(test-gpu "R6: values that do not fit the native types raise exn:fail:contract"
  (define X (rows-with 30 '(1 1)))
  (define two (labels-of (append '(0 1) (make-list 28 1)) 'int32))
  (check-exn exn:fail:contract? (lambda () (silhouette-score X two #:chunk-size (expt 2 63))))
  (check-exn exn:fail:contract? (lambda () (silhouette-samples X two #:chunk-size 2.5)))
  (define T (distinct-rows 30))
  (check-exn exn:fail:contract? (lambda () (trustworthiness T T #:n-neighbors (expt 2 63))))
  (check-exn exn:fail:contract? (lambda () (trustworthiness T T #:batch-size (expt 2 63)))))
