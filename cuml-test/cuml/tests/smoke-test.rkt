#lang racket/base

(require (only-in cuml exn:fail:cuml?)
         (only-in cuml/cluster/private/blobs blobs)
         (only-in racket/list remove-duplicates)
         (only-in rackunit check-equal? check-exn check-true)
         (only-in raft
                  device-matrix->list*
                  device-vector->list
                  dtype
                  exn:fail:raft-kind
                  exn:fail:raft?
                  shape)
         (only-in raft/tests/private/gpu test-gpu))

(define (labels-in y)
  (sort (remove-duplicates (device-vector->list y)) <))

(test-gpu "blobs, run as a job, fills raft arrays of the shape and dtype asked for"
  (for ([element-type (in-list '(float32 float64))])
    (define-values (X y) (blobs 60 3 #:centers 4 #:seed 3 #:dtype element-type))
    (check-equal? (list (shape X) (dtype X)) (list '(60 3) element-type))
    (check-equal? (list (shape y) (dtype y)) '((60) int32))
    (check-equal? (labels-in y) '(0 1 2 3))))

(test-gpu "blobs as a short call through with-array-views makes the job's data"
  (define-values (a a-labels) (blobs 40 2 #:seed 11))
  (define-values (b b-labels) (blobs 40 2 #:seed 11 #:job? #f))
  (check-equal? (device-matrix->list* b) (device-matrix->list* a))
  (check-equal? (device-vector->list b-labels) (device-vector->list a-labels)))

(define (check-cuml-error kind message thunk)
  (check-exn (lambda (e)
               (check-true (exn:fail:cuml? e))
               (check-true (exn:fail:raft? e))
               (check-equal? (exn:fail:raft-kind e) kind)
               (check-equal? (exn-message e) message)
               #t)
             thunk))

(test-gpu "a refused call raises exn:fail:cuml with the native message, job or not"
  (for ([job? (in-list '(#t #f))])
    (check-cuml-error 'logic
                      "blobs: n-clusters: expected at least 1, got 0"
                      (lambda () (blobs 10 2 #:centers 0 #:job? job?)))
    (check-cuml-error 'logic
                      "blobs: out: expected float32 or float64, got int32"
                      (lambda () (blobs 10 2 #:dtype 'int32 #:job? job?)))))
