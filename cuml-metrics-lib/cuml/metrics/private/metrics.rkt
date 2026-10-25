#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters define-codes)
         (only-in racket/match match-define)
         (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in "native.rkt"
                  cuml-metrics-adjusted-rand-score-job
                  cuml-metrics-contingency-matrix-job
                  cuml-metrics-silhouette-samples-job
                  cuml-metrics-silhouette-score-job
                  cuml-metrics-trustworthiness-job))

(provide (struct-out contingency-table)
         adjusted-rand-score
         adjusted-rand-score/who
         contingency-matrix
         contingency-matrix/who
         silhouette-samples
         silhouette-samples/who
         silhouette-score
         silhouette-score/who
         trustworthiness
         trustworthiness/who)

(struct contingency-table (counts row-labels column-labels))

(define-codes silhouette-metric-code
              [euclidean 0]
              [sqeuclidean 1]
              [manhattan 2]
              [cosine 3]
              [chebyshev 4])

(define-codes trustworthiness-metric-code [euclidean 0])

(define/who (contingency-matrix who
                                labels-true
                                labels-pred
                                #:resources [resources (current-device-resources)])
  (define out (deposit-for/who who resources))
  (cuml-metrics-contingency-matrix-job who resources labels-true labels-pred out)
  (apply contingency-table (deposit->list out)))

(define/who (adjusted-rand-score who
                                 labels-true
                                 labels-pred
                                 #:resources [resources (current-device-resources)])
  (cuml-metrics-adjusted-rand-score-job who resources labels-true labels-pred))

(define/who (silhouette-score who
                              X
                              labels
                              #:metric [metric 'euclidean]
                              #:chunk-size [chunk-size 40000]
                              #:resources [resources (current-device-resources)])
  (cuml-metrics-silhouette-score-job who
                                     resources
                                     X
                                     labels
                                     (silhouette-metric-code metric)
                                     chunk-size))

(define/who (silhouette-samples who
                                X
                                labels
                                #:metric [metric 'euclidean]
                                #:chunk-size [chunk-size 40000]
                                #:resources [resources (current-device-resources)])
  (define out (deposit-for/who who resources))
  (cuml-metrics-silhouette-samples-job who
                                       resources
                                       X
                                       labels
                                       out
                                       (silhouette-metric-code metric)
                                       chunk-size)
  (match-define (list samples) (deposit->list out))
  samples)

(define/who (trustworthiness who
                             X
                             X-embedded
                             #:n-neighbors [n-neighbors 5]
                             #:metric [metric 'euclidean]
                             #:batch-size [batch-size 512]
                             #:resources [resources (current-device-resources)])
  (cuml-metrics-trustworthiness-job who
                                    resources
                                    X
                                    X-embedded
                                    n-neighbors
                                    (trustworthiness-metric-code metric)
                                    batch-size))
