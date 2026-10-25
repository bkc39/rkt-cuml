#lang racket/base

(require (only-in cuml/core/private/define-who define/who))

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

(define/who (contingency-matrix who labels-true labels-pred #:resources [resources #f])
  (error 'unimplemented))

(define/who (adjusted-rand-score who labels-true labels-pred #:resources [resources #f])
  (error 'unimplemented))

(define/who (silhouette-score who
                              X
                              labels
                              #:metric [metric 'euclidean]
                              #:chunk-size [chunk-size 40000]
                              #:resources [resources #f])
  (error 'unimplemented))

(define/who (silhouette-samples who
                                X
                                labels
                                #:metric [metric 'euclidean]
                                #:chunk-size [chunk-size 40000]
                                #:resources [resources #f])
  (error 'unimplemented))

(define/who (trustworthiness who
                             X
                             X-embedded
                             #:n-neighbors [n-neighbors 5]
                             #:metric [metric 'euclidean]
                             #:batch-size [batch-size 512]
                             #:resources [resources #f])
  (error 'unimplemented))
