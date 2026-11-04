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
with a positive sum, and a zero weight deletes its row. @racket['auto] and
@racket['eig] solve the normal equations, @racket['svd] and
@racket['svd-jacobi] use a Jacobi SVD, @racket['qr] a QR factorisation and
@racket['svd-qr] a QR-based SVD. Each column, @racket[y] and the weights are
scaled by a power of two before the solve, so scaling a column by
2@superscript{k} scales its coefficient by 2@superscript{-k} exactly. A
column constant over the rows of positive weight (zero, without
@racket[fit-intercept]) gets coefficient 0. One column left always uses the
Jacobi SVD. With more columns left than rows of positive weight, less one
for @racket[fit-intercept], every column takes the same power and the Jacobi
SVD solves: the fit is the minimum-norm solution in float64, and in float32
without @racket[fit-intercept]; in float32 with it, the fit interpolates
@racket[y] but need not be minimum-norm. A solve that is not a
least-squares solution, as with collinear columns in float32, is refused;
@racket['qr] or float64 often fits. With weights and an intercept, fits above
a few thousand rows can differ in the last bits between runs, as in Python
cuML. @racket[X], @racket[y] and @racket[sample-weight] are not modified.

scikit-learn returns the minimum-norm solution for collinear columns too.
Python cuML does not scale, so a float32 column at 1e-5, a float32 @racket[y]
at 1e37 and a constant column under @racket['qr] give it wrong or NaN
coefficients.
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
weighted by @racket[sample-weight]. Errors name the generic.
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

@(close-eval ev)
