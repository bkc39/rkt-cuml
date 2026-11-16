#lang scribble/manual
@(require scribble/example
          (for-label cuml
                     racket/base
                     racket/contract/base
                     raft
                     threading))

@(define ev (make-base-eval #:lang 'racket/base '(require cuml raft threading)))
@(define int32-count @racket[(integer-in 1 2147483647)])

@title[#:tag "mbsgd"]{Mini-batch SGD}

@declare-exporting[cuml/linear-model cuml]

@examples[#:eval ev
(define X (list*->device-matrix '((0 1) (1 0) (2 1) (3 0) (4 1)) #:dtype 'float64))
(define y (list->device-vector '(1.0 2.0 4.0 5.0 7.0) #:dtype 'float64))
(define fitted (mbsgd-regressor-fit X y #:eta0 0.05 #:batch-size 5 #:shuffle #f))
(~> fitted
    mbsgd-regressor-result-coef
    device-vector->list
    (map (lambda (c) (real->decimal-string c 3)) _))
]

@defstruct*[mbsgd-regressor-result ([coef device-vector?] [intercept flonum?])]{
The iterate of @racket[mbsgd-regressor-fit]: one coefficient per column of the training
@racket[X], in its dtype, and the intercept, @racket[0.0] without @racket[fit-intercept].}

@defproc[(mbsgd-regressor-fit [X device-matrix?]
          [y device-vector?]
          [#:loss loss 'squared-loss 'squared-loss]
          [#:penalty penalty (or/c 'none 'l1 'l2 'elasticnet) 'l2]
          [#:alpha alpha (and/c rational? positive?) 1e-4]
          [#:l1-ratio l1-ratio (real-in 0 1) 0.15]
          [#:fit-intercept fit-intercept boolean? #t]
          [#:epochs epochs #,int32-count 1000]
          [#:tol tol (and/c rational? (>=/c 0)) 1e-3]
          [#:shuffle shuffle boolean? #t]
          [#:learning-rate learning-rate (or/c 'constant 'invscaling 'adaptive) 'constant]
          [#:eta0 eta0 (and/c rational? positive?) 1e-3]
          [#:power-t power-t (and/c rational? (>=/c 0)) 0.5]
          [#:batch-size batch-size #,int32-count 32]
          [#:n-iter-no-change n-iter-no-change #,int32-count 5]
          [#:random-state random-state (or/c #f (integer-in 0 4294967295)) #f]
          [#:params params (or/c #f hash?) #f]
          [#:resources resources device-resources? (current-device-resources)])
         mbsgd-regressor-result?]{
Minimises the mean squared error plus @racket[alpha] times the squared l2 norm, the l1 norm, or
@racket[l1-ratio] of the l1 norm and the rest of the squared l2 norm, by mini-batch SGD on
@racket[X] as given; the intercept, from centring, is not penalised. @racket[tol] 0 runs every
epoch. The result is the last iterate, not an optimum: steps depend on the scale of @racket[X], so
standardise it. Raises when the iterate is not finite or its objective F ends above
F₀·1.025 + A + 1e-12·F₀ plus round-off, F₀ the zero model's; lower @racket[eta0] then.
A = 3.5·k·@racket[eta0]·(@racket[alpha]·r)², k the number of columns plus 1 with
@racket[fit-intercept] and r the l1 share (1, @racket[l1-ratio] or 0), is capped at 4·F₀ for the
classifier only; the regressor's is uncapped (#134). Round-off is 10·ε²·mean(y²) for the
regressor in its dtype's machine epsilon ε. The gate sees only the objective, so it passes a fit
that ends at or below that bound yet predicts badly. A @racket[y] whose F₀ or round-off is not
finite, float64 |y| above about 1.3e154, is refused before the solve (rescale it). So is an
@racket[alpha], @racket[tol], @racket[eta0] or @racket[power-t] that overflows @racket[X]'s dtype,
an @racket[alpha] or @racket[eta0] that rounds to 0 in it, or an exact @racket[alpha] or
@racket[eta0] that rounds to 0.0. With @racket[random-state] an integer and @racket[shuffle], rows
are first permuted by that seed; @racket[#f] is cuML's fixed order. @racket[X] and @racket[y] are
not modified.
@examples[#:eval ev #:label #f
(eval:error (mbsgd-regressor-fit X y #:eta0 10.0 #:batch-size 5))
]}

@defproc[(mbsgd-regressor-predict [result mbsgd-regressor-result?] [X device-matrix?]
          [#:resources resources device-resources? (current-device-resources)])
         device-vector?]{
@racket[X] times @racket[coef] plus @racket[intercept].}

@defproc[(mbsgd-regressor-params [base (or/c #f hash?) #f]
          [#:loss loss 'squared-loss 'squared-loss]
          [#:penalty penalty (or/c 'none 'l1 'l2 'elasticnet) 'l2]
          [#:alpha alpha (and/c rational? positive?) 1e-4]
          [#:l1-ratio l1-ratio (real-in 0 1) 0.15]
          [#:fit-intercept fit-intercept boolean? #t]
          [#:epochs epochs #,int32-count 1000]
          [#:tol tol (and/c rational? (>=/c 0)) 1e-3]
          [#:shuffle shuffle boolean? #t]
          [#:learning-rate learning-rate (or/c 'constant 'invscaling 'adaptive) 'constant]
          [#:eta0 eta0 (and/c rational? positive?) 1e-3]
          [#:power-t power-t (and/c rational? (>=/c 0)) 0.5]
          [#:batch-size batch-size #,int32-count 32]
          [#:n-iter-no-change n-iter-no-change #,int32-count 5]
          [#:random-state random-state (or/c #f (integer-in 0 4294967295)) #f])
         (and/c hash? hash-eq? immutable?)]{
The hyperparameters of @racket[mbsgd-regressor-fit] and @racket[mbsgd-regressor], as
@racket[kmeans-params].}

@defstruct*[mbsgd-classifier-result ([coef device-vector?]
                                     [intercept flonum?]
                                     [classes list?]
                                     [label-kind any/c]
                                     [loss (or/c 'squared-loss 'log 'hinge)])]{
The iterate of @racket[mbsgd-classifier-fit], its two sorted @racket[classes] and the
@racket[loss] that picks the threshold. @racket[label-kind] is opaque, as in
@racket[logistic-regression-result].}

@defproc[(mbsgd-classifier-fit [X device-matrix?]
          [y (or/c device-vector? list?)]
          [#:loss loss (or/c 'hinge 'log 'squared-loss) 'hinge]
          [#:penalty penalty (or/c 'none 'l1 'l2 'elasticnet) 'l2]
          [#:alpha alpha (and/c rational? positive?) 1e-4]
          [#:l1-ratio l1-ratio (real-in 0 1) 0.15]
          [#:fit-intercept fit-intercept boolean? #t]
          [#:epochs epochs #,int32-count 1000]
          [#:tol tol (and/c rational? (>=/c 0)) 1e-3]
          [#:shuffle shuffle boolean? #t]
          [#:learning-rate learning-rate (or/c 'constant 'invscaling 'adaptive) 'constant]
          [#:eta0 eta0 (and/c rational? positive?) 1e-3]
          [#:power-t power-t (and/c rational? (>=/c 0)) 0.5]
          [#:batch-size batch-size #,int32-count 32]
          [#:n-iter-no-change n-iter-no-change #,int32-count 5]
          [#:random-state random-state (or/c #f (integer-in 0 4294967295)) #f]
          [#:params params (or/c #f hash?) #f]
          [#:resources resources device-resources? (current-device-resources)])
         mbsgd-classifier-result?]{
Two classes only. Fits the hinge loss on -1/+1 codes, or the log loss or squared error on 0/1
codes, of the sorted @racket[classes]. Under @racket[fit-intercept] the hinge and log fits learn
the intercept as a penalised coefficient on a column of ones, unlike sklearn's; squared-loss
centres. The hinge loss's @racket[tol] compares a sum over the rows, as cuML's does. Otherwise as
@racket[mbsgd-regressor-fit].
@examples[#:eval ev #:label #f
(define Xc (list*->device-matrix '((0 0) (1 0) (0 1) (3 3) (4 3) (3 4)) #:dtype 'float64))
(define fitted-c (mbsgd-classifier-fit Xc '(a a a b b b) #:eta0 0.05 #:batch-size 6))
(mbsgd-classifier-result-classes fitted-c)
(mbsgd-classifier-predict fitted-c (list*->device-matrix '((0 0) (4 4)) #:dtype 'float64))
]}

@defproc[(mbsgd-classifier-decision-function [result mbsgd-classifier-result?] [X device-matrix?]
          [#:resources resources device-resources? (current-device-resources)])
         device-vector?]{
@racket[X] times @racket[coef] plus @racket[intercept].}

@defproc[(mbsgd-classifier-predict [result mbsgd-classifier-result?] [X device-matrix?]
          [#:resources resources device-resources? (current-device-resources)])
         (or/c device-vector? list?)]{
The second class where the decision exceeds 0 (hinge, log) or 0.5 (squared-loss), else the first,
as @racket[y] was given. Python cuML thresholds the log loss at 0.5.}

@defproc[(mbsgd-classifier-params [base (or/c #f hash?) #f]
          [#:loss loss (or/c 'hinge 'log 'squared-loss) 'hinge]
          [#:penalty penalty (or/c 'none 'l1 'l2 'elasticnet) 'l2]
          [#:alpha alpha (and/c rational? positive?) 1e-4]
          [#:l1-ratio l1-ratio (real-in 0 1) 0.15]
          [#:fit-intercept fit-intercept boolean? #t]
          [#:epochs epochs #,int32-count 1000]
          [#:tol tol (and/c rational? (>=/c 0)) 1e-3]
          [#:shuffle shuffle boolean? #t]
          [#:learning-rate learning-rate (or/c 'constant 'invscaling 'adaptive) 'constant]
          [#:eta0 eta0 (and/c rational? positive?) 1e-3]
          [#:power-t power-t (and/c rational? (>=/c 0)) 0.5]
          [#:batch-size batch-size #,int32-count 32]
          [#:n-iter-no-change n-iter-no-change #,int32-count 5]
          [#:random-state random-state (or/c #f (integer-in 0 4294967295)) #f])
         (and/c hash? hash-eq? immutable?)]{
As @racket[mbsgd-regressor-params], with @racket[#:loss] defaulting to @racket['hinge].}

@section[#:tag "mbsgd-estimators"]{Mini-batch SGD estimators}

@declare-exporting[cuml]

@defproc[(mbsgd-regressor [params (or/c #f hash?) #f]
          [#:loss loss 'squared-loss 'squared-loss]
          [#:penalty penalty (or/c 'none 'l1 'l2 'elasticnet) 'l2]
          [#:alpha alpha (and/c rational? positive?) 1e-4]
          [#:l1-ratio l1-ratio (real-in 0 1) 0.15]
          [#:fit-intercept fit-intercept boolean? #t]
          [#:epochs epochs #,int32-count 1000]
          [#:tol tol (and/c rational? (>=/c 0)) 1e-3]
          [#:shuffle shuffle boolean? #t]
          [#:learning-rate learning-rate (or/c 'constant 'invscaling 'adaptive) 'constant]
          [#:eta0 eta0 (and/c rational? positive?) 1e-3]
          [#:power-t power-t (and/c rational? (>=/c 0)) 0.5]
          [#:batch-size batch-size #,int32-count 32]
          [#:n-iter-no-change n-iter-no-change #,int32-count 5]
          [#:random-state random-state (or/c #f (integer-in 0 4294967295)) #f]
          [#:resources resources (or/c #f device-resources?) #f])
         mbsgd-regressor?]{
As @racket[ridge]: @racket[fit] is @racket[mbsgd-regressor-fit] and needs @racket[y];
@racket[predict] is @racket[mbsgd-regressor-predict]; @racket[score] is @racket[r2-score]. A
@racket[sample-weight] is refused by @racket[fit].
@examples[#:eval ev #:label #f
(real->decimal-string (score (fit (mbsgd-regressor #:eta0 0.05 #:batch-size 5) X y) X y) 3)
]}

@defproc[(mbsgd-regressor? [v any/c]) boolean?]{Whether @racket[v] is a mini-batch SGD regressor.}

@defproc[(mbsgd-regressor-model [estimator mbsgd-regressor?]) (or/c #f mbsgd-regressor-result?)]{
The @racket[mbsgd-regressor-result] of the fit that made @racket[estimator], or @racket[#f].}

@defproc[(mbsgd-classifier [params (or/c #f hash?) #f]
          [#:loss loss (or/c 'hinge 'log 'squared-loss) 'hinge]
          [#:penalty penalty (or/c 'none 'l1 'l2 'elasticnet) 'l2]
          [#:alpha alpha (and/c rational? positive?) 1e-4]
          [#:l1-ratio l1-ratio (real-in 0 1) 0.15]
          [#:fit-intercept fit-intercept boolean? #t]
          [#:epochs epochs #,int32-count 1000]
          [#:tol tol (and/c rational? (>=/c 0)) 1e-3]
          [#:shuffle shuffle boolean? #t]
          [#:learning-rate learning-rate (or/c 'constant 'invscaling 'adaptive) 'constant]
          [#:eta0 eta0 (and/c rational? positive?) 1e-3]
          [#:power-t power-t (and/c rational? (>=/c 0)) 0.5]
          [#:batch-size batch-size #,int32-count 32]
          [#:n-iter-no-change n-iter-no-change #,int32-count 5]
          [#:random-state random-state (or/c #f (integer-in 0 4294967295)) #f]
          [#:resources resources (or/c #f device-resources?) #f])
         mbsgd-classifier?]{
As @racket[logistic-regression]: @racket[fit] is @racket[mbsgd-classifier-fit];
@racket[predict] and @racket[decision-function] are the @racket[mbsgd-classifier-] functions;
@racket[score] is @racket[accuracy-score]. A @racket[sample-weight] is refused by @racket[fit].}

@defproc[(mbsgd-classifier? [v any/c]) boolean?]{Whether @racket[v] is a mini-batch SGD classifier.}

@defproc[(mbsgd-classifier-model [estimator mbsgd-classifier?]) (or/c #f mbsgd-classifier-result?)]{
The @racket[mbsgd-classifier-result] of the fit that made @racket[estimator], or @racket[#f].}

@(close-eval ev)
