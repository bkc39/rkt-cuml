#lang racket/base

(require (only-in "../../core.rkt" exn:fail:cuml))

(provide logic-error)

(define (logic-error message . args)
  (raise (exn:fail:cuml (apply format message args) (current-continuation-marks) 'logic)))
