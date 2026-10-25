#lang scribble/manual
@(require scribble/example
          (for-label cuml
                     racket/base
                     racket/contract/base
                     raft
                     threading))

@(define ev (make-base-eval #:lang 'racket/base '(require cuml raft threading)))

@title[#:tag "decomposition"]{Decomposition}

@defmodule[cuml/decomposition #:no-declare]
@declare-exporting[cuml/decomposition cuml]

@section[#:tag "pca"]{Principal component analysis}

@examples[#:eval ev
(define X (list*->device-matrix '((4 0) (-4 0) (0 1) (0 -1) (0 0)) #:dtype 'float64))
(define fitted (pca-fit X))
(device-matrix->list* (pca-result-components fitted))
(map device-vector->list (list (pca-result-explained-variance fitted)
                               (pca-result-explained-variance-ratio fitted)
                               (pca-result-singular-values fitted)))
(device-matrix->list* (pca-transform fitted X))
]

@defstruct*[pca-result
            ([components device-matrix?]
             [explained-variance device-vector?]
             [explained-variance-ratio device-vector?]
             [singular-values device-vector?]
             [mean device-vector?]
             [noise-variance flonum?]
             [n-samples exact-positive-integer?]
             [whiten boolean?])]{
The result of @racket[pca-fit], every array in @racket[X]'s dtype: the
k×d @racket[components], orthonormal rows in decreasing order of the
variance each explains, each signed so that its first entry of largest
magnitude is positive; their @racket[explained-variance] (never negative)
and its share of the total; the @racket[singular-values], each the square
root of its variance times n − 1; the column @racket[mean]; the
@racket[noise-variance], the mean variance of the components not kept; and
the fit's @racket[whiten] flag, which the transforms read. A hand-built
result is checked only for dtypes and shapes, so non-finite,
negative-variance or non-orthonormal fields give unspecified, possibly
non-finite output.}

@defproc[(pca-fit [X device-matrix?]
                  [#:n-components n-components (or/c #f exact-positive-integer?) #f]
                  [#:whiten whiten boolean? #f]
                  [#:solver solver (or/c 'full 'jacobi) 'full]
                  [#:tol tol (and/c real? (>=/c 0)) 0.0]
                  [#:iterations iterations (integer-in 1 2147483647) 15]
                  [#:params params (or/c #f hash?) #f]
                  [#:resources resources device-resources? (current-device-resources)])
         pca-result?]{
The principal components of @racket[X], a row-major n×d @racket['float32]
or @racket['float64] matrix with at least two rows and columns, two distinct
rows and finite elements of magnitude at most √(@italic{max}/(4·n·d)),
@italic{max} the dtype's largest value. @racket[n-components] is at most
min(n, d); @racket[#f] keeps that many. @racket['jacobi] iterates at most
@racket[iterations] sweeps to tolerance @racket[tol], finite and at most the
float32 maximum (0 for machine precision), and stops silently when it runs
out; @racket['full] ignores both. Two fits of the same @racket[X] may differ
in the last bits.
@examples[#:eval ev #:label #f
(pca-result-noise-variance (pca-fit X #:n-components 1))
(eval:error (pca-fit X #:n-components 3))
(eval:error (pca-fit (list*->device-matrix '((1 2) (1 2)) #:dtype 'float64)))
]}

@defproc[(pca-transform [result pca-result?]
                        [X device-matrix?]
                        [#:resources resources device-resources? (current-device-resources)])
         device-matrix?]{
A new n×k matrix: @racket[X], less @racket[result]'s mean, times the
transposed components; when @racket[result] whitens, each column is then
divided by the square root of its variance (at least the dtype's epsilon),
so whitened training data has unit variance. @racket[X] needs the fit's
dtype and column count, a row, and finite elements of magnitude at most
@italic{max}/(4·d), times the smallest of those divisors when it is below 1.
@examples[#:eval ev #:label #f
(~> (pca-fit X #:whiten #t) (pca-transform X) device-matrix->list*)
(eval:error (pca-transform fitted (list*->device-matrix '((1 2 3)) #:dtype 'float64)))
]}

@defproc[(pca-inverse-transform [result pca-result?]
                                [X device-matrix?]
                                [#:resources resources device-resources? (current-device-resources)])
         device-matrix?]{
Maps n×k projections back: @racket[X], times each component's square root
of variance when @racket[result] whitens, times the components, plus the
mean. With every component kept, the round trip returns the data up to
rounding. @racket[X] needs the fit's dtype, k columns, a row, and finite
elements of magnitude at most @italic{max}/(4·k), over the largest square
root of variance when @racket[result] whitens and it is above 1.
@examples[#:eval ev #:label #f
(~> (pca-transform fitted X) (pca-inverse-transform fitted _) device-matrix->list*)
]}

@defproc[(pca-fit-transform [X device-matrix?]
                            [#:n-components n-components (or/c #f exact-positive-integer?) #f]
                            [#:whiten whiten boolean? #f]
                            [#:solver solver (or/c 'full 'jacobi) 'full]
                            [#:tol tol (and/c real? (>=/c 0)) 0.0]
                            [#:iterations iterations (integer-in 1 2147483647) 15]
                            [#:params params (or/c #f hash?) #f]
                            [#:resources resources device-resources? (current-device-resources)])
         (values pca-result? device-matrix?)]{
@racket[pca-fit], then @racket[pca-transform] of the same @racket[X] with
the fit it returns.}

@defproc[(pca-params [base (or/c #f hash?) #f]
                     [#:n-components n-components (or/c #f exact-positive-integer?) #f]
                     [#:whiten whiten boolean? #f]
                     [#:solver solver (or/c 'full 'jacobi) 'full]
                     [#:tol tol (and/c real? (>=/c 0)) 0.0]
                     [#:iterations iterations (integer-in 1 2147483647) 15])
         (and/c hash? hash-eq? immutable?)]{
The hyperparameters of @racket[pca-fit] and @racket[pca], as @racket[kmeans-params].
@examples[#:eval ev #:label #f
(hash-ref (pca-params #:solver 'jacobi) 'solver)
(eval:error (pca-params (hasheq 'copy #t)))
]}

@section[#:tag "tsvd"]{Truncated SVD}

@examples[#:eval ev
(define X (list*->device-matrix '((1 2 0) (2 4 1) (3 6 0) (4 8 1)) #:dtype 'float64))
(define fitted (tsvd-fit X #:n-components 2))
(device-matrix->list* (tsvd-result-components fitted))
(device-vector->list (tsvd-result-singular-values fitted))
(device-vector->list (tsvd-result-explained-variance-ratio fitted))
(device-matrix->list* (tsvd-transform fitted X))
]

@defstruct*[tsvd-result
            ([components device-matrix?]
             [explained-variance device-vector?]
             [explained-variance-ratio device-vector?]
             [singular-values device-vector?])]{
The result of @racket[tsvd-fit], every array in @racket[X]'s dtype: the
k×d @racket[components], the right singular vectors of @racket[X] as
orthonormal rows, each signed so that its first entry of largest magnitude
is positive; the population variance of each column of the transform, and
its share of the summed column variances of @racket[X] (0 when every row
is the same); and the k largest @racket[singular-values], descending.}

@defproc[(tsvd-fit [X device-matrix?]
                   [#:n-components n-components exact-positive-integer? 1]
                   [#:solver solver (or/c 'full 'jacobi) 'full]
                   [#:tol tol (and/c real? (>=/c 0)) 0.0]
                   [#:iterations iterations (integer-in 1 2147483647) 15]
                   [#:params params (or/c #f hash?) #f]
                   [#:resources resources device-resources? (current-device-resources)])
         tsvd-result?]{
The first @racket[n-components] singular triplets of @racket[X], a
row-major @racket['float32] or @racket['float64] matrix with at least two
rows, at least two columns and finite elements, without centring.
@racket[n-components] is at most the smaller of @racket[X]'s row and column
counts. The fit eigendecomposes XᵀX in @racket[X]'s dtype:
@racket['jacobi] iterates at most @racket[iterations] sweeps to tolerance
@racket[tol] (0 for machine precision); @racket['full] ignores both. A
singular value below about √ε·‖X‖ is not resolved and reads as a small
number or 0. A matrix whose squared Frobenius norm exceeds 2¹²⁵ (float32)
or 2¹⁰²¹ (float64) is refused, since XᵀX could overflow. On a matrix
whose rows are all close, the variance ratios are rounding noise. The same
@racket[X] gives the same bits.
@examples[#:eval ev #:label #f
(eval:error (tsvd-fit X #:n-components 4))
(eval:error (tsvd-fit X #:solver 'arpack))
]}

@defproc[(tsvd-transform [result tsvd-result?]
                         [X device-matrix?]
                         [#:resources resources device-resources? (current-device-resources)])
         device-matrix?]{
A new n×k matrix: @racket[X] times the transposed components.
@racket[X] needs the fit's dtype and column count, a row and finite
elements.}

@defproc[(tsvd-inverse-transform [result tsvd-result?]
                                 [X device-matrix?]
                                 [#:resources resources device-resources? (current-device-resources)])
         device-matrix?]{
A new n×d matrix: @racket[X], n×k, times the components. With every
column's component kept, the round trip returns the data up to rounding.
@examples[#:eval ev #:label #f
(define full (tsvd-fit X #:n-components 3))
(~> (tsvd-transform full X) (tsvd-inverse-transform full _) device-matrix->list*)
]}

@defproc[(tsvd-fit-transform [X device-matrix?]
                             [#:n-components n-components exact-positive-integer? 1]
                             [#:solver solver (or/c 'full 'jacobi) 'full]
                             [#:tol tol (and/c real? (>=/c 0)) 0.0]
                             [#:iterations iterations (integer-in 1 2147483647) 15]
                             [#:params params (or/c #f hash?) #f]
                             [#:resources resources device-resources? (current-device-resources)])
         (values tsvd-result? device-matrix?)]{
@racket[tsvd-fit], then @racket[tsvd-transform] of the same @racket[X],
with the same bits as the two calls.}

@defproc[(tsvd-params [base (or/c #f hash?) #f]
                      [#:n-components n-components exact-positive-integer? 1]
                      [#:solver solver (or/c 'full 'jacobi) 'full]
                      [#:tol tol (and/c real? (>=/c 0)) 0.0]
                      [#:iterations iterations (integer-in 1 2147483647) 15])
         (and/c hash? hash-eq? immutable?)]{
The hyperparameters of @racket[tsvd-fit] and @racket[truncated-svd], as
@racket[kmeans-params].
@examples[#:eval ev #:label #f
(hash-ref (tsvd-params #:n-components 3) 'n-components)
]}

@(close-eval ev)
