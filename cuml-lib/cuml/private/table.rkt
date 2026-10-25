#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in raft current-device-resources))

(provide table->device-matrix
         table->device-matrix/who)

(define/who (table->device-matrix who
                                  table
                                  #:columns [columns #f]
                                  #:dtype [element-type 'float32]
                                  #:layout [order 'row-major]
                                  #:resources [resources (current-device-resources)])
  (error 'unimplemented))
