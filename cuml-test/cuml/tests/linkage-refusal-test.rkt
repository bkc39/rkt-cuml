#lang racket/base

(require (only-in cuml/cluster single-linkage-fit)
         (only-in racket/function curry)
         (only-in racket/list make-list)
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
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "private/linkage-data.rkt" connectivities linked-lists))

(struct row (id message call))

(define (grid dt)
  (list*->device-matrix (for/list ([i (in-range 10)])
                          (list (add1 i) (* 2 i)))
                        #:dtype dt))

(define (with-row bad)
  (list*->device-matrix (append '((1 1) (2 3)) (list bad) '((4 4) (5 6)) (list '(+nan.0 0)))
                        #:dtype 'float32))

(define (norm-message got)
  (format "X: row 2: expected a squared norm of at most 4.253529586511731e+37, got ~a" got))

(define pairwise-message "X: pairwise connectivity holds at most 46340 rows, got 46341; use knn")

(define (fit-rows connectivity)
  (define X (grid 'float32))
  (define fit (curry single-linkage-fit #:connectivity connectivity))
  (list
   (row "L0 float64" "X: expected float32, got float64" (lambda () (fit (grid 'float64))))
   (row "L0 int32" "X: expected float32, got int32" (lambda () (fit (blank-matrix 10 2 'int32))))
   (row "L1"
        "connectivity: expected knn or pairwise"
        (lambda () (single-linkage-fit X #:connectivity 'mutual)))
   (row "L2" "metric: expected euclidean" (lambda () (fit X #:metric 'cosine)))
   (row "L2 l1" "metric: expected euclidean" (lambda () (fit X #:metric 'l1)))
   (row "L3 rank" "X: expected rank 2, got rank 1" (lambda () (fit (blank-vector 10 'float32))))
   (row "L3 layout"
        "X: expected row-major, got col-major 10x2"
        (lambda () (fit (blank-matrix 10 2 'float32 #:layout 'col-major))))
   (row "L4" "X: expected at least 1 row, got 0" (lambda () (fit (blank-matrix 0 2 'float32))))
   (row "L5" "X: expected at least 1 column, got 0" (lambda () (fit (blank-matrix 10 0 'float32))))
   (row "L7a" "n-clusters: expected at least 1, got 0" (lambda () (fit X #:n-clusters 0)))
   (row "L7a negative" "n-clusters: expected at least 1, got -2" (lambda () (fit X #:n-clusters -2)))
   (row "L7b" "X: 10 samples are fewer than the 11 clusters" (lambda () (fit X #:n-clusters 11)))
   (row "L8" "c: expected at most 2147483617, got 2147483618" (lambda () (fit X #:c 2147483618)))
   (row "L11 nan" (norm-message "nan") (lambda () (fit (with-row '(+nan.0 1)))))
   (row "L11 inf" (norm-message "inf") (lambda () (fit (with-row '(+inf.0 0)))))
   (row "L11 -inf" (norm-message "inf") (lambda () (fit (with-row '(0 -inf.0)))))
   (row "L11 1e20" (norm-message "1.0000000400817551e+40") (lambda () (fit (with-row '(1e20 0)))))
   (row "L11 2^63"
        (norm-message "8.507059173023462e+37")
        (lambda () (fit (with-row (list (expt 2 63) 0)))))
   (row "L0 before L1"
        "X: expected float32, got float64"
        (lambda () (single-linkage-fit (grid 'float64) #:connectivity 'mutual)))
   (row "L1 before L2"
        "connectivity: expected knn or pairwise"
        (lambda () (single-linkage-fit X #:connectivity 'mutual #:metric 'cosine)))
   (row "L2 before L3"
        "metric: expected euclidean"
        (lambda () (fit (blank-vector 10 'float32) #:metric 'cosine)))
   (row "L3 before L4" "X: expected rank 2, got rank 1" (lambda () (fit (blank-vector 0 'float32))))
   (row "L4 before L5"
        "X: expected at least 1 row, got 0"
        (lambda () (fit (blank-matrix 0 0 'float32))))
   (row "L5 before L7"
        "X: expected at least 1 column, got 0"
        (lambda () (fit (blank-matrix 10 0 'float32) #:n-clusters 0)))
   (row "L7a before L8"
        "n-clusters: expected at least 1, got 0"
        (lambda () (fit X #:n-clusters 0 #:c 2147483618)))
   (row "L7b before L8"
        "X: 10 samples are fewer than the 11 clusters"
        (lambda () (fit X #:n-clusters 11 #:c 2147483618)))
   (row "L8 before L11"
        "c: expected at most 2147483617, got 2147483618"
        (lambda () (fit (with-row '(+nan.0 1)) #:c 2147483618)))))

(test-gpu "single-linkage-fit refuses each bad argument with its message, under both connectivities"
  (for* ([connectivity (in-list connectivities)]
         [r (in-list (fit-rows connectivity))])
    (test-case (format "~a (~a)" (row-id r) connectivity)
      (check-cuml-error 'logic (format "single-linkage-fit: ~a" (row-message r)) (row-call r)))))

(test-gpu "R3: the pairwise row limit and the knn graph size are refused before the row scan"
  (check-cuml-error 'logic
                    (format "single-linkage-fit: ~a" pairwise-message)
                    (lambda ()
                      (single-linkage-fit (blank-matrix 46341 1 'float32) #:connectivity 'pairwise)))
  (define nan-column
    (list*->device-matrix (list* '(0) '(1) '(+nan.0) (make-list 46338 '(3))) #:dtype 'float32))
  (check-cuml-error 'logic
                    (format "single-linkage-fit: ~a" pairwise-message)
                    (lambda () (single-linkage-fit nan-column #:connectivity 'pairwise)))
  (check-cuml-error 'logic
                    (format "single-linkage-fit: ~a" (norm-message "nan"))
                    (lambda () (single-linkage-fit nan-column)))
  (check-cuml-error
   'logic
   (string-append "single-linkage-fit: X: 28000000 rows with 39 neighbours need 2184000000 knn graph"
                  " entries, more than 2147483647")
   (lambda () (single-linkage-fit (blank-matrix 28000000 1 'float32)))))

(test-gpu "R2: rows at the squared-norm boundary, and subnormal rows, fit"
  (define boundary
    (list*->device-matrix (list '(0 0)
                                (list (expt 2 62) (expt 2 62))
                                (list (- (expt 2 62)) (- (expt 2 62)))
                                '(1e-45 0)
                                '(3 4))
                          #:dtype 'float32))
  (for ([connectivity (in-list connectivities)])
    (test-case (symbol->string connectivity)
      (match (linked-lists boundary #:n-clusters 3 #:connectivity connectivity)
        [(list labels children)
         (check-equal? (length labels) 5)
         (check-equal? (length children) 4)]))))

(test-gpu "R4: values that do not fit the native types raise exn:fail:contract"
  (define X (grid 'float32))
  (check-exn exn:fail:contract? (lambda () (single-linkage-fit X #:c (expt 2 31))))
  (check-exn exn:fail:contract? (lambda () (single-linkage-fit X #:c (- -1 (expt 2 31)))))
  (check-exn exn:fail:contract? (lambda () (single-linkage-fit X #:n-clusters 2.5)))
  (check-exn exn:fail:contract? (lambda () (single-linkage-fit X #:n-clusters (expt 2 63)))))

(test-gpu "R4: released resources are refused under the procedure's own name"
  (define released (with-device-resources ([r (device-resources)]) r))
  (match (raised (lambda () (single-linkage-fit (grid 'float32) #:resources released)))
    [(? exn:fail:raft? e)
     (check-equal? (exn:fail:raft-kind e) 'logic)
     (check-equal? (exn-message e)
                   "single-linkage-fit: the device resources on device 0 were released")]
    [other (fail (format "expected an exn:fail:raft, got ~e" other))]))
