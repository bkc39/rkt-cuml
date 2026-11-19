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
standardise it.

Raises, advising a lower @racket[eta0], when the iterate is not finite or its objective ends above
the zero model's by more than a small slack, round-off of @racket[y], and, under l1 or elastic net,
an allowance for the oscillation about a zero optimum. The allowance grows with the columns,
@racket[eta0] and @racket[alpha] but is at most a quarter of the zero model's objective, a multiple
growing with the columns beyond 800. When that objective is itself round-off, as for a constant
@racket[y], an l1 or elastic-net regressor with @racket[fit-intercept] must instead fit the
constant: raises, advising a lower @racket[eta0] or @racket[alpha], unless its intercept is within
a millionth of the mean of @racket[y] and every coefficient within two penalty steps
(@racket[eta0] times @racket[alpha] times the l1 share) of 0; without @racket[fit-intercept] the
allowance is at most a quarter of the mean of @racket[y]'s squares. Also raises when the loss
without the penalty ends above the zero model's by more than the slack and, near round-off, up to as
much again, a fit that predicts worse than the zero model (the mean of @racket[y] with @racket[fit-intercept], 0 without). The gate sees only the objective and the loss, so it can pass a fit that predicts badly,
and refuse one that predicts well on a nearly constant @racket[y]. Raises, advising a rescale, before the solve when the zero model's objective of
@racket[y] is not finite, or when a sum the solver makes in @racket[X]'s dtype could overflow: the
sum of a column of @racket[X] (rescale @racket[X]), with @racket[fit-intercept] the sum of
@racket[y] (rescale @racket[y]), and the products of @racket[X] and the centred
@racket[y] in the gradient (rescale either; this bound is stricter than the solver needs when
@racket[batch-size] is below the rows). Also refused before the solve: an
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
centres. The hinge loss's @racket[tol] compares a sum over the rows, as cuML's does. The gate's
allowance is at most four times the zero model's objective, whatever the columns. Otherwise as
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
