#lang racket/base

(require (only-in racket/match match match*)
         (only-in raft
                  current-device-resources
                  device-array?
                  device-matrix
                  device-matrix?
                  device-vector
                  dtype
                  shape)
         (only-in "native.rkt"
                  cuml-cluster-kmeans-fit-job
                  cuml-cluster-kmeans-predict-job
                  cuml-cluster-kmeans-transform-job))

(provide (struct-out kmeans-result)
         kmeans-fit
         kmeans-predict
         kmeans-score
         kmeans-transform)

(struct kmeans-result (centroids labels inertia n-iter))

(define (init-rows init fallback)
  (match (and (device-matrix? init) (shape init))
    [(list rows _) rows]
    [_ fallback]))

(define (matrix-shape a)
  (match (and (device-array? a) (shape a))
    [(list rows cols) (list rows cols)]
    [_ #f]))

(define (fit-outputs X k resources)
  (match (matrix-shape X)
    [(list n d)
     #:when (and (exact-integer? k) (>= d 1) (<= 1 k n))
     (values (device-matrix k d #:dtype (dtype X) #:resources resources)
             (device-vector n #:dtype 'int32 #:resources resources))]
    [_ (values #f #f)]))

(define (seeding init oversampling-factor)
  (match init
    ['scalable-k-means++ (values 0 oversampling-factor #f)]
    ['k-means++ (values 0 0.0 #f)]
    ['random (values 1 oversampling-factor #f)]
    [(? device-matrix?) (values 2 oversampling-factor init)]
    [_ (values -1 oversampling-factor #f)]))

(define (n-init-runs n-init code)
  (match* (n-init code)
    [('auto 0) 1]
    [('auto _) 10]
    [(runs _) runs]))

(define (kmeans-fit X
                    #:init [init 'scalable-k-means++]
                    #:n-clusters [n-clusters (init-rows init 8)]
                    #:max-iter [max-iter 300]
                    #:tol [tol 1e-4]
                    #:n-init [n-init 'auto]
                    #:random-state [random-state 0]
                    #:oversampling-factor [oversampling-factor 2.0]
                    #:max-samples-per-batch [max-samples-per-batch 32768]
                    #:sample-weight [sample-weight #f]
                    #:resources [resources (current-device-resources)])
  (define-values (centroids labels) (fit-outputs X n-clusters resources))
  (define-values (code factor start) (seeding init oversampling-factor))
  (define-values (inertia n-iter)
    (cuml-cluster-kmeans-fit-job 'kmeans-fit
                                 resources
                                 X
                                 sample-weight
                                 start
                                 centroids
                                 labels
                                 n-clusters
                                 code
                                 max-iter
                                 tol
                                 (n-init-runs n-init code)
                                 factor
                                 max-samples-per-batch
                                 random-state))
  (kmeans-result centroids labels inertia n-iter))

(define (sample-rows X)
  (match (matrix-shape X)
    [(list n _) #:when (>= n 1) n]
    [_ #f]))

(define (kmeans-predict centroids
                        X
                        #:max-samples-per-batch [max-samples-per-batch 32768]
                        #:resources [resources (current-device-resources)])
  (define labels
    (match (sample-rows X)
      [#f #f]
      [n (device-vector n #:dtype 'int32 #:resources resources)]))
  (cuml-cluster-kmeans-predict-job 'kmeans-predict
                                   resources
                                   centroids
                                   X
                                   #f
                                   labels
                                   max-samples-per-batch)
  labels)

(define (kmeans-transform centroids
                          X
                          #:max-samples-per-batch [max-samples-per-batch 32768]
                          #:resources [resources (current-device-resources)])
  (define distances
    (match* ((sample-rows X) (sample-rows centroids))
      [((? exact-integer? n) (? exact-integer? k))
       (device-matrix n k #:dtype (dtype centroids) #:resources resources)]
      [(_ _) #f]))
  (cuml-cluster-kmeans-transform-job 'kmeans-transform
                                     resources
                                     centroids
                                     X
                                     distances
                                     max-samples-per-batch)
  distances)

(define (kmeans-score centroids
                      X
                      #:sample-weight [sample-weight #f]
                      #:max-samples-per-batch [max-samples-per-batch 32768]
                      #:resources [resources (current-device-resources)])
  (- (cuml-cluster-kmeans-predict-job 'kmeans-score
                                      resources
                                      centroids
                                      X
                                      sample-weight
                                      #f
                                      max-samples-per-batch)))
