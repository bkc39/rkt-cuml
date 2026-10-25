#lang scribble/manual
@(require scribble/example
          (for-label cuml
                     racket/base
                     racket/contract/base
                     raft
                     threading))

@(define ev (make-base-eval #:lang 'racket/base '(require cuml raft threading)))

@title[#:tag "preprocessing"]{Preprocessing}

@defmodule[cuml/preprocessing #:no-declare]
@declare-exporting[cuml/preprocessing cuml]

@section[#:tag "standard-scaler"]{Standard scaling}

@examples[#:eval ev
(define X (list*->device-matrix '((1 10) (3 10) (1 14) (3 14)) #:dtype 'float64))
(define fitted (standard-scaler-fit X))
(map device-vector->list (list (standard-scaler-result-mean fitted)
                               (standard-scaler-result-var fitted)
                               (standard-scaler-result-scale fitted)))
(device-matrix->list* (standard-scaler-transform fitted X))
]

@defstruct*[standard-scaler-result
            ([mean (or/c #f device-vector?)]
             [var (or/c #f device-vector?)]
             [scale (or/c #f device-vector?)]
             [n-samples exact-positive-integer?]
             [n-features exact-positive-integer?]
             [with-mean boolean?]
             [with-std boolean?])]{
The result of @racket[standard-scaler-fit]: each column's mean, population
variance and scale as @racket['float64] vectors whatever @racket[X]'s dtype;
the scale is the standard deviation, or 1 for a column indistinguishable from
a constant. @racket[mean] is @racket[#f] when neither flag is set, and
@racket[var] and @racket[scale] when @racket[with-std] is @racket[#f]. The
flags are the fit's, and say what the transforms apply.}

@defproc[(standard-scaler-fit [X device-matrix?]
                              [#:with-mean with-mean boolean? #t]
                              [#:with-std with-std boolean? #t]
                              [#:params params (or/c #f hash?) #f]
                              [#:resources resources device-resources? (current-device-resources)])
         standard-scaler-result?]{
Learns the column statistics of @racket[X], a row-major @racket['float32] or
@racket['float64] matrix with at least one row and one column and only finite
elements. Every statistic accumulates in @racket['float64], and the same
@racket[X] always gives the same bits. A column whose variance is within the
rounding error of a constant (at most n·ε·var + (n·mean·ε)², ε the
@racket['float64] epsilon) gets a scale of exactly 1, so it is only centred.
@racket[params], from @racket[standard-scaler-params], gives every
hyperparameter a keyword omits.
@examples[#:eval ev #:label #f
(define C (list*->device-matrix '((5 1) (5 3)) #:dtype 'float32))
(~> (standard-scaler-fit C) standard-scaler-result-scale device-vector->list)
(~> (standard-scaler-fit C) (standard-scaler-transform C) device-matrix->list*)
(eval:error (standard-scaler-fit (list*->device-matrix '((1 +inf.0)) #:dtype 'float64)))
]}

@defproc[(standard-scaler-transform [result standard-scaler-result?]
                                    [X device-matrix?]
                                    [#:resources resources device-resources? (current-device-resources)])
         device-matrix?]{
A new matrix of @racket[X]'s shape and dtype: each element less its column's
mean when @racket[result] centres, then divided by its scale when it scales,
both rounded to @racket[X]'s dtype first and computed in it. @racket[X] needs
@racket[result]'s number of columns, at least one row and finite elements,
and may have either dtype whatever the fit's.
@examples[#:eval ev #:label #f
(eval:error (standard-scaler-transform fitted (list*->device-matrix '((1)) #:dtype 'float64)))
]}

@defproc[(standard-scaler-inverse-transform [result standard-scaler-result?]
                                            [X device-matrix?]
                                            [#:resources resources device-resources? (current-device-resources)])
         device-matrix?]{
Undoes @racket[standard-scaler-transform]: each element times its column's
scale, then plus its mean, each step as the transform's. The round trip
returns @racket[X] up to rounding, not bit for bit.
@examples[#:eval ev #:label #f
(~> (standard-scaler-transform fitted X)
    (standard-scaler-inverse-transform fitted _)
    device-matrix->list*)
]}

@defproc[(standard-scaler-fit-transform [X device-matrix?]
                                        [#:with-mean with-mean boolean? #t]
                                        [#:with-std with-std boolean? #t]
                                        [#:params params (or/c #f hash?) #f]
                                        [#:resources resources device-resources? (current-device-resources)])
         (values standard-scaler-result? device-matrix?)]{
@racket[standard-scaler-fit], then @racket[standard-scaler-transform] of the
same @racket[X], with the same bits as the two calls.}

@defproc[(standard-scaler-params [base (or/c #f hash?) #f]
                                 [#:with-mean with-mean boolean? #t]
                                 [#:with-std with-std boolean? #t])
         (and/c hash? hash-eq? immutable?)]{
The hyperparameters of @racket[standard-scaler-fit] and @racket[standard-scaler],
as @racket[kmeans-params].
@examples[#:eval ev #:label #f
(hash-ref (standard-scaler-params #:with-mean #f) 'with-mean)
(eval:error (standard-scaler-params (hasheq 'copy #t)))
]}

@(close-eval ev)
