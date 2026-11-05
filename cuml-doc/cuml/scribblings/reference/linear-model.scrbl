#lang scribble/manual
@(require scribble/example
          (for-label cuml
                     racket/base
                     racket/contract/base
                     raft
                     threading))

@(define ev (make-base-eval #:lang 'racket/base '(require cuml raft threading)))

@title[#:tag "linear-model"]{Linear models}

@defmodule[cuml/linear-model #:no-declare]
@declare-exporting[cuml/linear-model cuml]

@section[#:tag "linear-regression"]{Linear regression}

@examples[#:eval ev
(define X (list*->device-matrix '((1 1) (1 2) (2 2) (2 3)) #:dtype 'float64))
(define y (list->device-vector '(6 8 9 11) #:dtype 'float64))
(define fitted (linear-regression-fit X y))
(~> fitted linear-regression-result-coef device-vector->list
    (map (lambda (c) (real->decimal-string c 6)) _))
(device-vector->list
 (linear-regression-predict fitted (list*->device-matrix '((3 5) (2 5)) #:dtype 'float64)))
]

@defstruct*[linear-regression-result ([coef device-vector?] [intercept flonum?])]{
The result of @racket[linear-regression-fit]: one coefficient per column of
the training @racket[X], in its dtype, and the intercept, exact in that
dtype and @racket[0.0] without @racket[fit-intercept].}

@defproc[(linear-regression-fit [X device-matrix?]
                                [y device-vector?]
                                [#:sample-weight sample-weight (or/c #f device-vector?) #f]
                                [#:algorithm algorithm (or/c 'auto 'eig 'svd 'svd-jacobi 'qr 'svd-qr) 'auto]
                                [#:fit-intercept fit-intercept boolean? #t]
                                [#:params params (or/c #f hash?) #f]
                                [#:resources resources device-resources? (current-device-resources)])
         linear-regression-result?]{
Fits ordinary least squares to @racket[y] on the columns of @racket[X], a
row-major @racket['float32] or @racket['float64] matrix of finite values with
at least two rows. @racket[y] and @racket[sample-weight] are vectors of
@racket[X]'s dtype with one finite element per row; weights are at least 0
with a positive sum. A zero weight deletes its row, and so does a weight of
at most 2@superscript{−150}·4@superscript{k} (float32) or
2@superscript{−1075}·4@superscript{k} (float64), 4@superscript{k} being the
largest weight's power of two rounded up to a power of 4.

@racket['auto] and @racket['eig] solve the normal equations, @racket['svd]
and @racket['svd-jacobi] use a Jacobi SVD, @racket['qr] a QR factorisation
and @racket['svd-qr] a QR-based SVD; each column, @racket[y] and the weights
are first scaled by powers of two.

A column constant over the rows of positive weight (zero, without
@racket[fit-intercept]) gets coefficient 0. With more non-constant columns
than rows of positive weight, less one for @racket[fit-intercept], the Jacobi
SVD gives the minimum-norm fit, except in float32 with
@racket[fit-intercept], where the fit interpolates @racket[y] but generally
is not minimum-norm.

The fit is refused when the residual's weighted inner product with some
centred column exceeds 2@superscript{-8} (float32) or 2@superscript{-18}
(float64) times that column's weighted spread times @racket[y]'s, as with
collinear columns in float32; @racket['qr] or float64 often fits. A
coefficient fixed only by rows of low weight w is accurate to about
u·√(w@subscript{max}/w), u being the dtype's unit roundoff, and the refusal
does not see its loss.

Weighted fits with an intercept can differ in the last bits between runs
from about 2000 rows, as in Python cuML; unweighted fits, and fits without an
intercept, repeat bitwise. @racket[X], @racket[y] and @racket[sample-weight]
are not modified.
@examples[#:eval ev #:label #f
(define Xc (list*->device-matrix '((0 1) (1 1) (2 1) (3 1)) #:dtype 'float64))
(define yc (list->device-vector '(1 3 5 7) #:dtype 'float64))
(~> (linear-regression-fit Xc yc) linear-regression-result-coef device-vector->list)
(define w (list->device-vector '(1 2 3 4) #:dtype 'float64))
(~> (linear-regression-fit X (list->device-vector '(6 8 10 11) #:dtype 'float64)
                           #:sample-weight w #:algorithm 'qr)
    linear-regression-result-intercept
    (real->decimal-string 6))
(eval:error (linear-regression-fit X (list->device-vector '(1 2 3) #:dtype 'float64)))
]}

@defproc[(linear-regression-predict [result linear-regression-result?]
                                    [X device-matrix?]
                                    [#:resources resources device-resources? (current-device-resources)])
         device-vector?]{
@racket[X] times @racket[result]'s coefficients plus its intercept, one
element per row. @racket[X] is a row-major matrix of the fit's dtype with
finite values, at least one row and one column per coefficient.
@examples[#:eval ev #:label #f
(eval:error (linear-regression-predict fitted (list*->device-matrix '((1 2 3)) #:dtype 'float64)))
]}

@defproc[(linear-regression-params [base (or/c #f hash?) #f]
                                   [#:algorithm algorithm (or/c 'auto 'eig 'svd 'svd-jacobi 'qr 'svd-qr) 'auto]
                                   [#:fit-intercept fit-intercept boolean? #t])
         (and/c hash? hash-eq? immutable?)]{
The hyperparameters of @racket[linear-regression-fit] and
@racket[linear-regression], as @racket[kmeans-params].}

@section[#:tag "linear-regression-estimator"]{Linear regression estimator}

@declare-exporting[cuml]

@defproc[(linear-regression [params (or/c #f hash?) #f]
                            [#:algorithm algorithm (or/c 'auto 'eig 'svd 'svd-jacobi 'qr 'svd-qr) 'auto]
                            [#:fit-intercept fit-intercept boolean? #t]
                            [#:resources resources (or/c #f device-resources?) #f])
         linear-regression?]{
An unfitted linear regression with the hyperparameters
@racket[(linear-regression-params params ...)]: an estimator, a predictor
and a scorer. @racket[fit] is @racket[linear-regression-fit] and needs
@racket[y]; @racket[predict] is @racket[linear-regression-predict];
@racket[score] is @racket[r2-score] of @racket[y] against the predictions,
weighted by @racket[sample-weight]; in float32 it can count a row whose
weight @racket[fit] counted as zero. Errors name the generic.
@examples[#:eval ev #:label #f
(define e (fit (linear-regression) X y))
(real->decimal-string (score e X y) 6)
(eval:error (fit (linear-regression) X))
]}

@defproc[(linear-regression? [v any/c]) boolean?]{
Whether @racket[v] is a linear regression estimator.}

@defproc[(linear-regression-model [estimator linear-regression?])
         (or/c #f linear-regression-result?)]{
The @racket[linear-regression-result] of the fit that made
@racket[estimator], or @racket[#f] when it is unfitted.}

@section[#:tag "ridge"]{Ridge regression}

@declare-exporting[cuml/linear-model cuml]

@examples[#:eval ev
(define (decimals k cs) (map (lambda (c) (real->decimal-string c k)) cs))
(define fitted-r (ridge-fit X y #:alpha 1.0))
(~> fitted-r ridge-result-coef device-vector->list (decimals 6 _))
(device-vector->list
 (ridge-predict fitted-r (list*->device-matrix '((3 5) (2 5)) #:dtype 'float64)))
]

@defstruct*[ridge-result ([coef device-vector?] [intercept flonum?])]{
The result of @racket[ridge-fit], as @racket[linear-regression-result].}

@defproc[(ridge-fit [X device-matrix?]
                    [y device-vector?]
                    [#:sample-weight sample-weight (or/c #f device-vector?) #f]
                    [#:alpha alpha (and/c rational? (>=/c 0)) 1.0]
                    [#:solver solver (or/c 'auto 'eig 'svd) 'auto]
                    [#:fit-intercept fit-intercept boolean? #t]
                    [#:params params (or/c #f hash?) #f]
                    [#:resources resources device-resources? (current-device-resources)])
         ridge-result?]{
Minimises Σ@subscript{i} w@subscript{i}(y@subscript{i} − b@subscript{0} − x@subscript{i}·b)² +
@racket[alpha]·‖b‖², as scikit-learn's and Python cuML's @tt{Ridge}: @racket[alpha]
is not scaled by the row count or the weights' sum, and the intercept is not
penalised. @racket[X], @racket[y] and @racket[sample-weight] are as for
@racket[linear-regression-fit].

@racket['auto] and @racket['eig] eigendecompose XᵀWX; @racket['svd] takes a
QR-based SVD of X. Every column and @racket[alpha] are first scaled by powers of
two. An @racket[alpha] of at most u times the largest column's weighted spread
Σ@subscript{i}w@subscript{i}(x@subscript{ij} − x̄@subscript{j})² (x̄@subscript{j}
0 without @racket[fit-intercept]), u being the dtype's unit roundoff, is taken as 0:
that fit is @racket[linear-regression-fit]'s with @racket['eig] (@racket['auto],
@racket['eig]) or @racket['svd-qr] (@racket['svd]), refusals included. Such an
@racket[alpha] can still shrink a column of much smaller spread; scale the columns
alike, or fit in float64. A column
constant over the rows of positive weight (zero, without @racket[fit-intercept])
gets coefficient 0.

The fit is refused when the solve misses the ridge solution by more than
2@superscript{-8} (float32) or 2@superscript{-18} (float64) in gradient cosine, as
when X is ill-conditioned for @racket[alpha] (float32 @racket['eig] on columns of
very different scales); @racket['svd] or float64 often fits. It is refused too when
@racket[alpha] scaled with X passes the dtype's range. Coefficients are accurate to
about u·(σ@subscript{max}² + @racket[alpha])/(σ@subscript{min}² + @racket[alpha]).

Weighted fits with an intercept can differ in the last bits between runs from
about 2000 rows, as in Python cuML; unweighted fits, and fits without an
intercept, repeat bitwise. @racket[X], @racket[y] and @racket[sample-weight] are
not modified.
@examples[#:eval ev #:label #f
(~> (ridge-fit X y #:alpha 0) ridge-result-coef device-vector->list (decimals 6 _))
(~> (ridge-fit X y #:alpha 1e6) ridge-result-coef device-vector->list (decimals 9 _))
(eval:error (ridge-fit X y #:alpha -1))
(eval:error (ridge-fit X y #:solver 'lsmr))
]}

@defproc[(ridge-predict [result ridge-result?] [X device-matrix?]
                        [#:resources resources device-resources? (current-device-resources)])
         device-vector?]{
As @racket[linear-regression-predict], for a @racket[ridge-result].}

@defproc[(ridge-params [base (or/c #f hash?) #f]
                       [#:alpha alpha (and/c rational? (>=/c 0)) 1.0]
                       [#:solver solver (or/c 'auto 'eig 'svd) 'auto]
                       [#:fit-intercept fit-intercept boolean? #t])
         (and/c hash? hash-eq? immutable?)]{
The hyperparameters of @racket[ridge-fit] and @racket[ridge], as @racket[kmeans-params].}

@section[#:tag "ridge-estimator"]{Ridge estimator}

@declare-exporting[cuml]

@defproc[(ridge [params (or/c #f hash?) #f]
                [#:alpha alpha (and/c rational? (>=/c 0)) 1.0]
                [#:solver solver (or/c 'auto 'eig 'svd) 'auto]
                [#:fit-intercept fit-intercept boolean? #t]
                [#:resources resources (or/c #f device-resources?) #f])
         ridge?]{
An unfitted ridge regression with the hyperparameters
@racket[(ridge-params params ...)], as @racket[linear-regression]: @racket[fit] is
@racket[ridge-fit] and needs @racket[y], @racket[predict] is @racket[ridge-predict],
and @racket[score] is the weighted @racket[r2-score].
@examples[#:eval ev #:label #f
(real->decimal-string (score (fit (ridge #:alpha 0.5) X y) X y) 6)
(eval:error (fit (ridge) X))
]}

@defproc[(ridge? [v any/c]) boolean?]{Whether @racket[v] is a ridge estimator.}

@defproc[(ridge-model [estimator ridge?]) (or/c #f ridge-result?)]{
The @racket[ridge-result] of the fit that made @racket[estimator], or @racket[#f].}

@(close-eval ev)
