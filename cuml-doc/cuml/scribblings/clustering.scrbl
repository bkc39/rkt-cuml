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
                     [#:eps eps (and/c real? (>/c 0) (</c +inf.0)) 0.5]
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
cosine similarity. A row with a NaN or an infinity is noise.
@racket[max-mbytes-per-batch] bounds, in units of 10^6
bytes, the memory of each batch of the pairwise distances; @racket[#f] or
@racket[0] sizes the batch from the device's memory. The outputs live on
@racket[resources].

@bold{Cosine boundary.} The fit runs on a unit-row copy of @racket[X] in its
dtype. A pair at distance exactly @racket[eps] is included when those rows are
exact, as one-hot rows are, and every pair is included at @racket[eps] 2 or
more. Otherwise a pair within a relative 10^-5 of @racket[eps], at 1024
@racket['float32] columns, may fall on either side, and the tolerance grows
with the number of columns.

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
       matrix with a row and a column, or with @racket['precomputed] is not
       square;}
 @item{@racket[out-dtype] or @racket[metric] is not one of its names;}
 @item{@racket[X] has more than 2^31-1 rows, or too many elements or rows
       for @racket['int32] labels: @tt{X: expected at most 2147483647 rows,
       got 2147483648}, @tt{X: 46341x46341 elements do not fit int32 labels;
       use int64}, @tt{X: 2147483646 rows do not fit int32 labels; use
       int64};}
 @item{@racket[sample-weight] is not a vector of @racket[X]'s dtype with one
       element per row;}
 @item{@racket[eps] is NaN, not greater than 0 in @racket[X]'s dtype, or
       past its largest value: @tt{eps: expected at most
       3.4028234663852886e+38, got 1e+300};}
 @item{@racket[min-samples] is less than 1;}
 @item{@racket[max-mbytes-per-batch] is negative or more than
       18446744073709;}
 @item{a batch would hold no row, where cuML divides by zero: @racket[n]
       rows with @racket[s]-byte labels need
       ⌈(@racket[n](2@racket[s]+2)+2@racket[s])/10^6⌉, @tt{max-mbytes-per-batch:
       expected at least 2 for 150000 rows, got 1}, and with @racket[#f] or
       @racket[0] a large @racket[X] can leave the automatic size too small,
       @tt{max-mbytes-per-batch: the automatic size holds no batch of 69000
       rows; pass at least 2};}
 @item{with @racket['cosine], a row of @racket[X] is all zeros:
       @tt{X: row 2 is zero, so no cosine distance}.}]
An @racket[X] or @racket[sample-weight] that is not a device array, or
@racket[resources] that were released, raises raft's @racket[exn:fail:raft]
naming the procedure, and a number argument that does not fit the native type,
such as @racket[#:min-samples 2.5], raises @racket[exn:fail:contract].

@section[#:tag "single-linkage"]{Single linkage}

@examples[#:eval ev
(define P (list*->device-matrix '((0) (1) (3) (7) (15)) #:dtype 'float32))
(match-define (single-linkage-result labels children) (single-linkage-fit P #:n-clusters 3))
(device-vector->list labels)
(device-matrix->list* children)
]

@defstruct*[single-linkage-result ([labels device-vector?] [children device-matrix?])]{

The result of @racket[single-linkage-fit], both @racket['int32].
@racket[children] is the dendrogram, an (n−1)×2 matrix: row @racket[i] is the
@racket[i]th merge, in non-decreasing weight as cuML computes it in
@racket['float32], and names the two nodes it joins, the smaller first. Node
@racket[j] below n is sample @racket[j], and node n+@racket[i] is the cluster
that row @racket[i] forms. Merges at equal distances come in an order cuML
chooses.

@racket[labels] gives each sample its cluster when the dendrogram is cut into
@racket[n-clusters] clusters: their roots, in descending order of node, are
labelled 0 to k−1, so cluster 0 formed last and single samples take the
highest labels.}

@defproc[(single-linkage-fit [X device-matrix?]
                             [#:n-clusters n-clusters exact-positive-integer? 2]
                             [#:metric metric 'euclidean 'euclidean]
                             [#:connectivity connectivity (or/c 'knn 'pairwise) 'knn]
                             [#:c c (integer-in -2147483648 2147483617) 15]
                             [#:resources resources device-resources? (current-device-resources)])
         single-linkage-result?]{

Clusters the rows of @racket[X], a row-major @racket['float32] matrix, by
single linkage under the euclidean distance: each step merges the two closest
clusters, the distance between two clusters being that of their closest
samples.

With @racket['pairwise], every pairwise distance is held on the device, so
@racket[X] can have at most 46340 rows, and the hierarchy is exact in the
@racket['float32] distances below.

@racket['knn] is approximate, the hierarchy of a sparse graph in which each
sample keeps its k = min(n, max(2, ⌊log₂ n⌋ + @racket[c])) nearest samples,
its components joined; at k = n it is @racket['pairwise]'s graph, but merges
at nearly equal distances can pair or order differently, since the two compute
distances differently in @racket['float32]. When distances tie,
@racket['knn] can give a different hierarchy from one run to the next.

The distances are ‖x‖²+‖y‖²−2x·y in @racket['float32]: centre @racket[X] if
it lies far from the origin. The outputs live on @racket[resources].

@racket[P]'s cut is the same under @racket['pairwise]:

@examples[#:eval ev #:label #f
(~> (single-linkage-fit P #:n-clusters 3 #:connectivity 'pairwise)
    single-linkage-result-labels
    device-vector->list)
]

With one cluster per sample, sample @racket[i] has label n−1−@racket[i]:

@examples[#:eval ev #:label #f
(~> (single-linkage-fit P #:n-clusters 5) single-linkage-result-labels device-vector->list)
]

@racket[X] must be @racket['float32]:

@examples[#:eval ev #:label #f
(eval:error (single-linkage-fit (list*->device-matrix '((0) (1)) #:dtype 'float64)))
]}

@bold{Errors.} @racket[single-linkage-fit] raises @racket[exn:fail:cuml] of
kind @racket['logic], leaving nothing allocated, when:
@itemlist[
 @item{@racket[X] is not a row-major @racket['float32] matrix with a row and
       a column;}
 @item{@racket[connectivity] or @racket[metric] is not one of its names;}
 @item{@racket[X] has more than 2^31-1 elements;}
 @item{@racket[n-clusters] is less than 1 or more than the number of rows;}
 @item{@racket[c] is more than 2147483617;}
 @item{with @racket['pairwise], @racket[X] has more than 46340 rows;}
 @item{with @racket['knn], the graph's 2nk entries would pass 2^31-1;}
 @item{a row holds a NaN or an infinity, or its squared norm passes 2^125.}]
cuML's own failures, such as a @racket['knn] graph it could not join (raise
@racket[c]), raise @racket[exn:fail:cuml] naming the procedure, with nothing
allocated, and running out of device memory has kind @racket['oom]. An
@racket[X] that is not a device array, or @racket[resources] that were
released, raises raft's @racket[exn:fail:raft] naming the procedure, and a
number argument that does not fit the native type, such as
@racket[#:n-clusters 2.5], raises @racket[exn:fail:contract].

@section[#:tag "spectral"]{Spectral clustering}

@examples[#:eval ev
(define G
  (list*->device-matrix (for*/list ([centre (in-list '(0 10 20))]
                                    [i (in-range 8)])
                          (list (+ centre (* 0.5 (quotient i 4))) (* 0.5 (remainder i 4))))
                        #:dtype 'float32))
(~> (spectral-clustering-fit G #:n-clusters 3 #:n-neighbors 5)
    spectral-clustering-result-labels
    device-vector->list)
]

@defstruct*[spectral-clustering-result ([labels device-vector?])]{

The result of @racket[spectral-clustering-fit] and
@racket[spectral-clustering-fit-graph]: each sample's cluster, @racket['int32],
from 0 to k−1. The labels are k-means' over the spectral embedding, so their
numbering depends on @racket[random-state] and a label can go unused: compare
partitions, not label values.}

@defproc[(spectral-clustering-fit [X device-matrix?]
                                  [#:n-clusters n-clusters exact-positive-integer? 8]
                                  [#:n-components n-components (or/c #f exact-positive-integer?) #f]
                                  [#:random-state random-state (integer-in 0 18446744073709551615) 0]
                                  [#:n-neighbors n-neighbors (and/c exact-integer? (>=/c 2)) 10]
                                  [#:n-init n-init (integer-in 1 2147483647) 10]
                                  [#:eigen-tol eigen-tol (or/c 'auto (and/c real? (>=/c 0))) 'auto]
                                  [#:resources resources device-resources? (current-device-resources)])
         spectral-clustering-result?]{

Clusters the rows of @racket[X], a row-major @racket['float32] matrix with at
least 4 rows, from the normalized Laplacian of the graph that joins each
sample to its min(@racket[n-neighbors], n−1) nearest samples, itself
included. The first @racket[n-components] eigenvectors, by default
min(@racket[n-clusters], ⌊(n−1)/3⌋), are clustered by k-means with
@racket[n-init] runs, seeded by @racket[random-state]. An explicit
@racket[n-components] must lie in [1, ⌊(n−1)/3⌋], and more components than
clusters can blur the partition.

The eigensolver stops when its residual is at most @racket[eigen-tol]
(@racket['auto] is 0) or after 10n iterations, and never reports that it did
not converge. Near cluster boundaries the labels can differ from one run to
the next at the default @racket[eigen-tol]. Graphs whose components are
cliques, and rows repeated more often than @racket[n-neighbors], can fail with
``eigensolver couldn't converge''. The labels live on @racket[resources].}

@defproc[(spectral-clustering-fit-graph [rows device-vector?]
                                        [cols device-vector?]
                                        [vals device-vector?]
                                        [n-samples (integer-in 4 2147483647)]
                                        [#:n-clusters n-clusters exact-positive-integer? 8]
                                        [#:n-components n-components (or/c #f exact-positive-integer?) #f]
                                        [#:random-state random-state (integer-in 0 18446744073709551615) 0]
                                        [#:n-init n-init (integer-in 1 2147483647) 10]
                                        [#:eigen-tol eigen-tol (or/c 'auto (and/c real? (>=/c 0))) 'auto]
                                        [#:resources resources device-resources? (current-device-resources)])
         spectral-clustering-result?]{

Clusters @racket[n-samples] samples from an affinity graph given as
coordinates: entry @racket[i] joins sample @racket[(rows i)] to sample
@racket[(cols i)] with weight @racket[(vals i)]. @racket[rows] and
@racket[cols] are @racket['int32] and @racket[vals] is @racket['float32], all
of one length. The graph must be symmetric, without self-loops, with finite
weights of at least 0. Duplicate entries add, and a sample with no edges gets
some label. Otherwise as @racket[spectral-clustering-fit].

@racket[G]'s groups, joined within each group to the samples at most two away:

@examples[#:eval ev #:label #f
(define-values (rows cols)
  (for*/lists (rows cols)
              ([i (in-range 24)]
               [j (in-range 24)]
               #:when (and (not (= i j))
                           (= (quotient i 8) (quotient j 8))
                           (<= (abs (- i j)) 2)))
    (values i j)))
(~> (spectral-clustering-fit-graph (list->device-vector rows #:dtype 'int32)
                                   (list->device-vector cols #:dtype 'int32)
                                   (list->device-vector (map (lambda (_) 1.0) rows) #:dtype 'float32)
                                   24
                                   #:n-clusters 3)
    spectral-clustering-result-labels
    device-vector->list)
]

A self-loop is refused:

@examples[#:eval ev #:label #f
(eval:error
 (spectral-clustering-fit-graph (list->device-vector '(0 1 2) #:dtype 'int32)
                                (list->device-vector '(1 1 0) #:dtype 'int32)
                                (list->device-vector '(1.0 1.0 1.0) #:dtype 'float32)
                                4
                                #:n-clusters 2))
]}

@bold{Errors.} @racket[spectral-clustering-fit] and
@racket[spectral-clustering-fit-graph] raise @racket[exn:fail:cuml] of kind
@racket['logic], leaving nothing allocated, when:
@itemlist[
 @item{@racket[X] is not a row-major @racket['float32] matrix with at least 4
       rows and a column, or @racket[rows], @racket[cols] and @racket[vals] are
       not vectors of their dtypes and of one length;}
 @item{@racket[X] has more than 2^31-1 elements, its kNN graph would pass
       2^31-1 entries, or the graph's entries and samples together pass
       2^31-1;}
 @item{@racket[n-samples] is less than 4 or more than 2^31-1;}
 @item{@racket[n-clusters] is less than 1 or more than the number of samples;}
 @item{@racket[n-components] is outside [1, ⌊(n−1)/3⌋], @racket[n-neighbors]
       is less than 2, @racket[n-init] is less than 1, or @racket[eigen-tol] is
       negative, NaN or past the largest @racket['float32];}
 @item{a row of @racket[X] holds a NaN or an infinity, or its squared norm
       passes 2^125;}
 @item{an entry's index is outside [0, @racket[n-samples]), the entry is a
       self-loop, or its weight is negative or not finite: the first such
       entry is named;}
 @item{no weight is positive.}]
cuML's own failures, such as an eigensolver that could not converge, raise
@racket[exn:fail:cuml] of kind @racket['generic] naming the procedure, with nothing allocated, and
running out of device memory has kind @racket['oom]. An argument that is not a
device array, or @racket[resources] that were released, raises raft's
@racket[exn:fail:raft] naming the procedure, and a number argument that does
not fit the native type, such as @racket[#:n-init 2.5], raises
@racket[exn:fail:contract].

@(close-eval ev)
