#lang scribble/manual
@(require scribble/example
          (for-label cuml
                     racket/base
                     racket/contract/base
                     racket/match
                     raft))

@(define ev (make-base-eval #:lang 'racket/base '(require cuml raft racket/match threading)))

@title[#:tag "clustering"]{Clustering}

@defmodule[cuml/cluster #:no-declare]
@declare-exporting[cuml/cluster cuml]

Each procedure takes a @racket[device-matrix] of samples, one per row, and
returns its results on the same @racket[device-resources].

@section[#:tag "kmeans"]{k-means}

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
                     [#:max-samples-per-batch max-samples-per-batch exact-nonnegative-integer? 32768]
                     [#:sample-weight sample-weight (or/c #f device-vector?) #f]
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

@defproc[(kmeans-predict [centroids device-matrix?]
                         [X device-matrix?]
                         [#:max-samples-per-batch max-samples-per-batch exact-nonnegative-integer? 32768]
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
                           [#:max-samples-per-batch max-samples-per-batch exact-nonnegative-integer? 32768]
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
                       [#:max-samples-per-batch max-samples-per-batch exact-nonnegative-integer? 32768]
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

@section[#:tag "dbscan"]{DBSCAN}

@examples[#:eval ev
(define P (list*->device-matrix '((0) (1) (2) (4) (6) (7) (8) (20)) #:dtype 'float64))
(match-define (dbscan-result labels core) (dbscan-fit P #:eps 2 #:min-samples 4))
(device-vector->list labels)
(device-vector->list core)
]

Sample 3 borders both clusters and joins cluster 0, and sample 7 is noise.

@defstruct*[dbscan-result ([labels device-vector?] [core-sample-indices device-vector?])]{

The result of @racket[dbscan-fit]: each sample's cluster in @racket[labels],
@racket[-1] for noise and @racket[0] to @racket[(- k 1)] otherwise, numbered by
each cluster's smallest core sample; and the indices of the core samples in
ascending order. Both are of the @racket[out-dtype].}

@defproc[(dbscan-fit [X device-matrix?]
                     [#:eps eps (and/c real? (>/c 0)) 0.5]
                     [#:min-samples min-samples exact-positive-integer? 5]
                     [#:metric metric (or/c 'euclidean 'cosine 'precomputed) 'euclidean]
                     [#:out-dtype out-dtype (or/c 'int32 'int64) 'int32]
                     [#:max-mbytes-per-batch max-mbytes-per-batch
                      (or/c #f exact-nonnegative-integer?) #f]
                     [#:sample-weight sample-weight (or/c #f device-vector?) #f]
                     [#:resources resources device-resources? (current-device-resources)])
         dbscan-result?]{

Clusters the rows of @racket[X], a row-major @racket['float32] or
@racket['float64] matrix, by density. A sample's neighbourhood is the samples
at distance at most @racket[eps], itself included. A sample is core when its
neighbourhood holds @racket[min-samples] samples, or that much
@racket[sample-weight] (a vector of @racket[X]'s dtype, passed to cuML as it
is). Core samples within @racket[eps] of each other share a cluster. A border
sample, not core but near a core sample, joins one of its core neighbours'
clusters, and which one may depend on the batching. Every other sample is
noise.

With @racket['precomputed], @racket[X] is a square matrix of distances, row
@racket[i] from sample @racket[i]. @racket['cosine] distance is one minus the
cosine similarity, and @racket[X] is not modified: the fit runs as euclidean
DBSCAN on a copy of @racket[X] scaled to unit rows in @racket[X]'s dtype. A
row with a NaN or an infinity is noise.
@racket[max-mbytes-per-batch] bounds, in units of 10^6
bytes, the memory of each batch of the pairwise distances; @racket[#f] or
@racket[0] sizes the batch from the device's memory. The outputs live on
@racket[resources].

Weights count towards @racket[min-samples]:

@examples[#:eval ev #:label #f
(define w (list->device-vector '(4 1 1 1 1 1 1 1) #:dtype 'float64))
(~> (dbscan-fit P #:eps 2 #:min-samples 4 #:sample-weight w)
    dbscan-result-core-sample-indices
    device-vector->list)
]

A precomputed matrix of the distances gives the same labels:

@examples[#:eval ev #:label #f
(define points '(0 1 2 4 6 7 8 20))
(define D
  (list*->device-matrix (for/list ([a (in-list points)])
                          (for/list ([b (in-list points)])
                            (abs (- a b))))
                        #:dtype 'float64))
(~> (dbscan-fit D #:eps 2 #:min-samples 4 #:metric 'precomputed)
    dbscan-result-labels
    device-vector->list)
]

The labels and indices take the @racket[out-dtype]:

@examples[#:eval ev #:label #f
(dtype (dbscan-result-labels (dbscan-fit P #:eps 2 #:min-samples 4 #:out-dtype 'int64)))
]

@racket[eps] must be finite and greater than 0:

@examples[#:eval ev #:label #f
(eval:error (dbscan-fit P #:eps 0))
]}

@bold{Errors.} @racket[dbscan-fit] raises @racket[exn:fail:cuml] of kind
@racket['logic], leaving nothing allocated, when:
@itemlist[
 @item{@racket[X] is not a row-major @racket['float32] or @racket['float64]
       matrix with at least one row and one column, or with
       @racket['precomputed] is not square;}
 @item{@racket[out-dtype] is not @racket['int32] or @racket['int64], or
       @racket[metric] is not one of the three names;}
 @item{@racket[X] has more than 2^31-1 rows, at either @racket[out-dtype]:
       @tt{X: expected at most 2147483647 rows, got 2147483648};}
 @item{@racket['int32] labels cannot index @racket[X]'s elements, past
       2^31-1 of them, or its rows, past 2^31-3: @tt{X: 46341x46341 elements
       do not fit int32 labels; use int64}, @tt{X: 2147483646 rows do not fit
       int32 labels; use int64};}
 @item{@racket[sample-weight] is not a vector of @racket[X]'s dtype with one
       element per row;}
 @item{@racket[eps] is not greater than 0 in @racket[X]'s dtype, NaN
       included, or is more than the dtype's largest value, @racket[+inf.0]
       included: @tt{eps: expected at most 3.4028234663852886e+38, got
       1e+300} for @racket['float32];}
 @item{@racket[min-samples] is less than 1;}
 @item{@racket[max-mbytes-per-batch] is negative, or more than
       18446744073709, past which its bytes overflow;}
 @item{the batch holds no row, where cuML would kill the process with a
       division by zero. A batch over @racket[n] rows with labels of
       @racket[s] bytes needs at least
       ⌈(@racket[n](2@racket[s]+2)+2@racket[s])/10^6⌉, so 150000 rows with
       @racket['int32] labels need 2: @tt{max-mbytes-per-batch: expected at
       least 2 for 150000 rows, got 1}. With @racket[#f] or @racket[0], cuML
       sizes the batch from 80% of the device's total memory less the size
       of @racket[X], which a large enough @racket[X] leaves too small:
       @tt{max-mbytes-per-batch: the automatic size holds no batch of 69000
       rows; pass at least 2};}
 @item{with @racket['cosine], a row of @racket[X] is all zeros,
       @racket[-0.0] included: @tt{X: row 2 is zero, so no cosine
       distance}.}]
An @racket[X] or @racket[sample-weight] that is not a device array, or
@racket[resources] that were released, raises raft's @racket[exn:fail:raft]
naming the procedure, and a number argument that does not fit the native type,
such as @racket[#:min-samples 2.5], raises @racket[exn:fail:contract].

@(close-eval ev)
