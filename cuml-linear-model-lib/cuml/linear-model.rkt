#lang racket/base

(require "linear-model/private/linear-regression.rkt")

(provide (except-out (all-from-out "linear-model/private/linear-regression.rkt")
                     linear-regression-fit/who
                     linear-regression-predict/who))
