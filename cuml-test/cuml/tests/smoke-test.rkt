#lang racket/base

(require (only-in cuml exn:fail:cuml?)
         (only-in cuml/cluster/private/blobs blobs blobs/short-call)
         (only-in racket/list remove-duplicates)
         (only-in rackunit check-equal? check-pred)
         (only-in raft device-matrix->list* device-vector->list dtype exn:fail:raft-kind shape)
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
  (define-values (b b-labels) (blobs/short-call 40 2 #:seed 11))
  (check-equal? (device-matrix->list* b) (device-matrix->list* a))
  (check-equal? (device-vector->list b-labels) (device-vector->list a-labels)))

(define (raised thunk)
  (with-handlers ([exn:fail? values])
    (thunk)
    #f))

(define (check-cuml-error kind message thunk)
  (define e (raised thunk))
  (check-pred exn:fail:cuml? e)
  (check-equal? (list (exn:fail:raft-kind e) (exn-message e)) (list kind message)))

(test-gpu "a refused call raises exn:fail:cuml with the native message, job or not"
  (for ([make-blobs (in-list (list blobs blobs/short-call))])
    (check-cuml-error 'logic
                      "blobs: n-clusters: expected at least 1, got 0"
                      (lambda () (make-blobs 10 2 #:centers 0)))
    (check-cuml-error 'logic
                      "blobs: out: expected float32 or float64, got int32"
                      (lambda () (make-blobs 10 2 #:dtype 'int32)))))
