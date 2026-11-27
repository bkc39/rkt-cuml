#lang racket/base

(require "svm/private/svc.rkt")

(provide (except-out (all-from-out "svm/private/svc.rkt")
                     svc-decision-function/who
                     svc-fit/who
                     svc-predict/who))
