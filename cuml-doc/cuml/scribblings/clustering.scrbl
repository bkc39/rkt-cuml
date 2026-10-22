#lang scribble/manual
@(require scribble/example
          (for-label cuml
                     racket/base
                     racket/contract/base
                     racket/match
                     raft))

@(define ev (make-base-eval #:lang 'racket/base '(require cuml raft racket/match threading)))

@title[#:tag "clustering"]{k-means clustering}

@defmodule[cuml/cluster #:no-declare]
@declare-exporting[cuml/cluster cuml]

Each procedure takes a @racket[device-matrix] of samples, one per row, and
returns its results on the same @racket[device-resources].

@examples[#:eval ev
(define X
  (list*->device-matrix '((0 0) (0 2) (2 0) (2 2) (10 10) (10 12) (12 10) (12 12))
                        #:dtype 'float64))
(match-define (kmeans-result centroids labels inertia n-iter)
  (kmeans-fit X #:n-clusters 2 #:random-state 1))
(list inertia n-iter)
(sort (device-matrix->list* centroids) < #:key car)
]

@defstruct*[kmeans-result ([centroids device-matrix?]
                           [labels device-vector?]
                           [inertia (and/c flonum? (>=/c 0.0))]
                           [n-iter exact-nonnegative-integer?])]{

The result of @racket[kmeans-fit]: the @racket[(n-clusters n-features)]
@racket[centroids] in @racket[X]'s dtype; each sample's cluster in
@racket[labels], an @racket['int32] vector of values from @racket[0] to
@racket[(- n-clusters 1)]; the @racket[inertia] of @racket[X] on the
centroids, the weighted sum of squared distances from each sample to its
nearest centroid, with the weights rescaled to sum to the number of samples;
and @racket[n-iter], the Lloyd iterations run.}

@defproc[(kmeans-fit [X device-matrix?]
                     [#:init init
                      (or/c 'scalable-k-means++ 'k-means++ 'random device-matrix?)
                      'scalable-k-means++]
                     [#:n-clusters n-clusters exact-positive-integer?
                      (if (device-matrix? init) (car (shape init)) 8)]
                     [#:max-iter max-iter exact-nonnegative-integer? 300]
                     [#:tol tol (and/c real? (>/c 0)) 1e-4]
                     [#:n-init n-init (or/c 'auto exact-positive-integer?) 'auto]
                     [#:random-state random-state exact-nonnegative-integer? 0]
                     [#:oversampling-factor oversampling-factor (and/c real? (>=/c 0)) 2.0]
                     [#:params params (or/c #f hash?) #f]
                     [#:sample-weight sample-weight (or/c #f device-vector?) #f]
                     [#:max-samples-per-batch max-samples-per-batch (integer-in 0 2147483647)
                      (current-batch-samples)]
                     [#:resources resources device-resources? (current-device-resources)])
         kmeans-result?]{

Clusters the rows of @racket[X], a row-major @racket['float32] or
@racket['float64] matrix, into @racket[n-clusters] clusters by Lloyd's
algorithm, stopping after @racket[max-iter] iterations or when the centroids
move less than @racket[tol].

The seeding is @racket[init]:
@itemlist[
 @item{@racket['scalable-k-means++] seeds with k-means||, sampling
       @racket[oversampling-factor] × @racket[n-clusters] points per round,
       and with an @racket[oversampling-factor] of 0 runs the sequential
       k-means++;}
 @item{@racket['k-means++] is its sequential form, and ignores
       @racket[oversampling-factor];}
 @item{@racket['random] picks rows of @racket[X];}
 @item{a matrix of @racket[X]'s dtype gives the starting centroids, one per
       row, and is not modified.}]

The fit runs @racket[n-init] times from different seedings and keeps the run
of lowest inertia; @racket['auto] is 1 for the k-means++ seedings and 10
otherwise. @racket[random-state] seeds the seeding. @racket[sample-weight], a
vector of @racket[X]'s dtype, weighs each sample; zero, negative and NaN
weights are passed to cuML as they are. @racket[max-samples-per-batch] bounds
the samples processed at once. The outputs live on @racket[resources].
@racket[params], from @racket[kmeans-params], gives every hyperparameter a keyword omits.
The batch size changes results no more than repeating the call does.

@examples[#:eval ev #:label #f
(define start (list*->device-matrix '((1 0) (11 11)) #:dtype 'float64))
(~> (kmeans-fit X #:init start #:max-iter 1) kmeans-result-centroids device-matrix->list*)
]

With @racket[#:max-iter 0] the result is the seeding:

@examples[#:eval ev #:label #f
(~> (kmeans-fit X #:init start #:max-iter 0) kmeans-result-centroids device-matrix->list*)
]

Weights pull the centroids:

@examples[#:eval ev #:label #f
(define w (list->device-vector '(1 3 1 3 1 1 1 1) #:dtype 'float64))
(~> (kmeans-fit X #:init start #:max-iter 1 #:sample-weight w)
    kmeans-result-centroids
    device-matrix->list*)
]

There must be at least as many samples as clusters:

@examples[#:eval ev #:label #f
(eval:error (kmeans-fit X #:n-clusters 9))
]}

@defproc[(kmeans-params [base (or/c #f hash?) #f]
                        [#:init init
                         (or/c 'scalable-k-means++ 'k-means++ 'random device-matrix?)
                         'scalable-k-means++]
                        [#:n-clusters n-clusters exact-positive-integer?
                         (if (device-matrix? init) (car (shape init)) 8)]
                        [#:max-iter max-iter exact-nonnegative-integer? 300]
                        [#:tol tol (and/c real? (>/c 0)) 1e-4]
                        [#:n-init n-init (or/c 'auto exact-positive-integer?) 'auto]
                        [#:random-state random-state exact-nonnegative-integer? 0]
                        [#:oversampling-factor oversampling-factor (and/c real? (>=/c 0)) 2.0])
         (and/c hash? hash-eq? immutable?)]{

The hyperparameters of @racket[kmeans-fit] and @racket[kmeans], every key present: @racket[base], or with
@racket[#f] the defaults, with each given keyword's value. From a @racket[base] nothing is recomputed, so
an omitted @racket[n-clusters] keeps @racket[base]'s. Only keys are checked: a @racket[base] with another
key, or without one, raises @racket[exn:fail:cuml] of kind @racket['logic]; values are checked by the fit.

@examples[#:eval ev #:label #f
(define tight (kmeans-params #:tol 1e-6 #:n-init 3))
(hash-ref (kmeans-params tight #:n-init 5) 'n-init)
(for/list ([k (in-range 1 4)])
  (kmeans-result-inertia (kmeans-fit X #:params tight #:n-clusters k)))
(eval:error (kmeans-params (hasheq 'colour 'red)))
]}

@defparam[current-batch-samples samples (integer-in 0 2147483647) #:value 32768]{

The samples k-means processes at once, read when @racket[kmeans-fit], @racket[kmeans-predict],
@racket[kmeans-transform] and @racket[kmeans-score] are called without @racket[#:max-samples-per-batch],
and by a @racket[kmeans] estimator built without it. It changes results no more than repeating the call
does.

@examples[#:eval ev #:label #f
(equal? (~> (parameterize ([current-batch-samples 3]) (kmeans-fit X #:n-clusters 2 #:random-state 1))
            kmeans-result-labels
            device-vector->list)
        (~> (kmeans-fit X #:n-clusters 2 #:random-state 1) kmeans-result-labels device-vector->list))
]}

@defproc[(kmeans-predict [centroids device-matrix?]
                         [X device-matrix?]
                         [#:max-samples-per-batch max-samples-per-batch (integer-in 0 2147483647)
                          (current-batch-samples)]
                         [#:resources resources device-resources? (current-device-resources)])
         device-vector?]{

The nearest of the @racket[centroids] to each row of @racket[X], as an
@racket['int32] vector of row indices of @racket[centroids].

@examples[#:eval ev #:label #f
(define cells (kmeans-result-centroids (kmeans-fit X #:init start)))
(device-vector->list
 (kmeans-predict cells (list*->device-matrix '((1 1) (11 12)) #:dtype 'float64)))
]}

@defproc[(kmeans-transform [centroids device-matrix?]
                           [X device-matrix?]
                           [#:max-samples-per-batch max-samples-per-batch (integer-in 0 2147483647)
                          (current-batch-samples)]
                           [#:resources resources device-resources? (current-device-resources)])
         device-matrix?]{

The @racket[(n n-clusters)] matrix of @emph{squared} Euclidean distances from
each row of @racket[X] to each centroid, as cuML computes them, in the
centroids' dtype.

@examples[#:eval ev #:label #f
(~> (kmeans-transform cells X) device-matrix->list* car)
]}

@defproc[(kmeans-score [centroids device-matrix?]
                       [X device-matrix?]
                       [#:sample-weight sample-weight (or/c #f device-vector?) #f]
                       [#:max-samples-per-batch max-samples-per-batch (integer-in 0 2147483647)
                          (current-batch-samples)]
                       [#:resources resources device-resources? (current-device-resources)])
         flonum?]{

The negated inertia of @racket[X] on the @racket[centroids], with the weights
rescaled to sum to the number of samples. On the data a result @racket[r]
was fitted on, it is @racket[(- (kmeans-result-inertia r))].

@examples[#:eval ev #:label #f
(list (kmeans-score cells X) (- inertia))
]}

@bold{Errors.} Each procedure raises @racket[exn:fail:cuml] of kind @racket['logic], before
the GPU is touched and leaving nothing allocated, when:
@itemlist[
 @item{@racket[X] or @racket[centroids] is not a row-major @racket['float32]
       or @racket['float64] matrix with at least one row and one column, or
       has more elements than a 32-bit index reaches;}
 @item{@racket[X]'s dtype or column count differs from the centroids';}
 @item{@racket[n-clusters] is less than 1 or more than @racket[X]'s rows;}
 @item{@racket[sample-weight] is not a vector of @racket[X]'s dtype with one
       element per row;}
 @item{@racket[init] is not one of the names above, or a matrix of another
       dtype, another shape than @racket[(n-clusters n-features)], or not
       row-major;}
 @item{@racket[max-iter] or @racket[max-samples-per-batch] is negative, or
       @racket[n-init] is less than 1;}
 @item{@racket[tol] is not greater than 0, or @racket[oversampling-factor]
       is negative or NaN with any @racket[init] but @racket['k-means++],
       which ignores it;}
 @item{@racket[kmeans-transform]'s @racket[(n n-clusters)] distances have
       more elements than a 32-bit index reaches, such as a
       @racket[(100000 4)] @racket[X] on @racket[(50000 4)] centroids:
       @tt{distances: 100000x50000 elements do not fit the index type}.}]
An @racket[X] or @racket[centroids] that is not a device array, or
@racket[resources] that were released, raises raft's @racket[exn:fail:raft]
naming the procedure, and a number argument that is not exact, such as
@racket[#:n-init 2.5], raises @racket[exn:fail:contract].

@(close-eval ev)
