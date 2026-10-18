#lang racket/base

#|review: ignore|#

;; whole-module: struct-out needs the structure-type bindings only-in strips
(require "cluster/private/kmeans.rkt"
         (only-in "cluster/private/native.rkt")) ;; noqa

(provide (struct-out kmeans-result)
         kmeans-fit
         kmeans-predict
         kmeans-score
         kmeans-transform)
