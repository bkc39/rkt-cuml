#lang racket/base

(require "linear-model/private/linear-regression.rkt"
         "linear-model/private/ridge.rkt")

(provide (except-out (all-from-out "linear-model/private/linear-regression.rkt")
                     linear-regression-fit/who
                     linear-regression-predict/who)
         (except-out (all-from-out "linear-model/private/ridge.rkt") ridge-fit/who ridge-predict/who))
