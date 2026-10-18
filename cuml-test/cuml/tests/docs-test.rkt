#lang racket/base

(require (only-in cuml
                  exn:fail:cuml
                  exn:fail:cuml?
                  kmeans-fit
                  kmeans-predict
                  kmeans-result
                  kmeans-result-centroids
                  kmeans-score
                  kmeans-transform)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in raft
                  device-matrix->list*
                  device-vector->list
                  exn:fail:raft
                  exn:fail:raft-kind
                  exn:fail:raft?
                  list*->device-matrix
                  list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error))

(test-case "the manual: a raft handler catches exn:fail:cuml and reads its kind"
  (check-equal? (with-handlers ([exn:fail:raft? exn:fail:raft-kind])
                  (raise (exn:fail:cuml "my-binding: n-clusters: expected at least 1, got 0"
                                        (current-continuation-marks)
                                        'logic)))
                'logic))

(test-case "the manual: exn:fail:cuml is both predicates' and keeps its kind"
  (define e (exn:fail:cuml "my-binding: out of memory" (current-continuation-marks) 'out-of-memory))
  (check-equal? (list (exn:fail:cuml? e) (exn:fail:raft? e) (exn:fail:raft-kind e))
                '(#t #t out-of-memory)))

(test-case "the manual: a raft failure is not a cuML failure"
  (check-equal? (exn:fail:cuml?
                 (exn:fail:raft "device-resources: no device 7" (current-continuation-marks) 'logic))
                #f))

(define (grid)
  (list*->device-matrix '((0 0) (0 2) (2 0) (2 2) (10 10) (10 12) (12 10) (12 12)) #:dtype 'float64))

(define (start)
  (list*->device-matrix '((1 0) (11 11)) #:dtype 'float64))

(define (chapter-fit)
  (kmeans-fit (grid) #:n-clusters 2 #:random-state 1))

(test-gpu "the manual: the chapter's fit finds the two cells of the grid"
  (match-define (kmeans-result centroids _ inertia _) (chapter-fit))
  (check-equal? inertia 16.0)
  (check-equal? (sort (device-matrix->list* centroids) < #:key car) '((1.0 1.0) (11.0 11.0))))

(test-gpu "the manual: one step from a given start is the cell means, with n-clusters inferred"
  (check-equal? (device-matrix->list* (kmeans-result-centroids (kmeans-fit (grid) #:init (start) #:max-iter 1)))
                '((1.0 1.0) (11.0 11.0))))

(test-gpu "the manual: #:max-iter 0 returns the start"
  (check-equal? (device-matrix->list* (kmeans-result-centroids (kmeans-fit (grid) #:init (start) #:max-iter 0)))
                '((1.0 0.0) (11.0 11.0))))

(test-gpu "the manual: weights pull the centroids"
  (define w (list->device-vector '(1 3 1 3 1 1 1 1) #:dtype 'float64))
  (check-equal? (device-matrix->list*
                 (kmeans-result-centroids (kmeans-fit (grid) #:init (start) #:max-iter 1 #:sample-weight w)))
                '((1.0 1.5) (11.0 11.0))))

(test-gpu "the manual: more clusters than samples is refused"
  (check-cuml-error 'logic
                    "kmeans-fit: X: 8 samples are fewer than the 9 clusters"
                    (lambda () (kmeans-fit (grid) #:n-clusters 9))))

(test-gpu "the manual: predict, transform and score on the cells of the grid"
  (match-define (kmeans-result _ _ inertia _) (chapter-fit))
  (define cells (kmeans-result-centroids (kmeans-fit (grid) #:init (start))))
  (check-equal? (device-vector->list
                 (kmeans-predict cells (list*->device-matrix '((1 1) (11 12)) #:dtype 'float64)))
                '(0 1))
  (match-define (list* first-row _) (device-matrix->list* (kmeans-transform cells (grid))))
  (check-equal? first-row '(2.0 242.0))
  (check-equal? (list (kmeans-score cells (grid)) (- inertia)) '(-16.0 -16.0)))
