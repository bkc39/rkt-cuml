#lang scribble/manual
@(require scribble/example
          (for-label cuml
                     racket/base
                     racket/contract/base
                     raft
                     threading))

@(define ev (make-base-eval #:lang 'racket/base '(require cuml raft threading)))

@title[#:tag "hdbscan"]{HDBSCAN}

@declare-exporting[cuml/cluster cuml]

@examples[#:eval ev
(define X
  (list*->device-matrix (for*/list ([centre (in-list '(0 20 40))]
                                    [i (in-range 12)])
                          (list (+ centre (* 0.5 (quotient i 4))) (* 0.5 (remainder i 4))))
                        #:dtype 'float32))
(define result (hdbscan-fit X #:min-cluster-size 5))
(device-vector->list (hdbscan-result-labels result))
(~> (hdbscan-approximate-predict result
                                 (list*->device-matrix '((0.5 0.5) (40.5 1) (10 30))
                                                       #:dtype 'float32))
    hdbscan-prediction-labels
    device-vector->list)
]

@defstruct*[hdbscan-result ([labels device-vector?]
                            [probabilities device-vector?]
                            [cluster-persistence device-vector?]
                            [condensed-tree hdbscan-condensed-tree?]
                            [single-linkage-tree hdbscan-linkage-tree?]
                            [minimum-spanning-tree hdbscan-spanning-tree?]
                            [model native-model?])
            #:omit-constructor]{

The result of @racket[hdbscan-fit]. @racket[labels] is @racket['int64]: −1 is
noise, and 0 to k−1 number the selected clusters in ascending order of their
condensed-tree node. @racket[probabilities] is @racket['float32] in [0, 1],
the strength of each sample's membership, 0 on noise.
@racket[cluster-persistence] holds each selected cluster's stability, and can
vary between runs in its last bits. @racket[model] is a
@tech{native model} of kind @racket['hdbscan]: it holds the C++ state the
prediction procedures read, until it is released or collected. Every array
stays valid after a release.}

@defstruct*[hdbscan-condensed-tree ([parents device-vector?]
                                    [children device-vector?]
                                    [lambdas device-vector?]
                                    [sizes device-vector?]
                                    [selected-clusters device-vector?])
            #:omit-constructor]{

The condensed cluster tree, one entry per edge: @racket[parents],
@racket[children] and @racket[sizes] are @racket['int64] and @racket[lambdas]
is @racket['float32]. Nodes 0 to n−1 are the samples and nodes n and up are
clusters, with n the root. Label i is cluster node n + @racket[(selected-clusters i)].}

@defstruct*[hdbscan-linkage-tree ([children device-matrix?]
                                  [distances device-vector?]
                                  [sizes device-vector?])
            #:omit-constructor]{

The single-linkage tree of the mutual reachability graph: n−1 merges, in
order of @racket['float32] @racket[distances]. Each row of the @racket['int64]
(n−1)×2 @racket[children] is a sorted pair of the nodes merged, numbered as
for @racket[single-linkage-result], and @racket[sizes] counts the samples
under each merge.}

@defstruct*[hdbscan-spanning-tree ([sources device-vector?]
                                   [destinations device-vector?]
                                   [weights device-vector?])
            #:omit-constructor]{

The minimum spanning tree of the mutual reachability graph: n−1 edges, with
@racket['int64] endpoints and @racket['float32] @racket[weights] equal to the
linkage tree's distances.}

@defstruct*[hdbscan-prediction ([labels device-vector?]
                                [probabilities device-vector?])
            #:omit-constructor]{

The result of @racket[hdbscan-approximate-predict]: an @racket['int64] label
and a @racket['float32] probability for each point, as in
@racket[hdbscan-result].}

@defproc[(hdbscan-fit [X device-matrix?]
                      [#:min-cluster-size min-cluster-size (and/c exact-integer? (>=/c 2)) 5]
                      [#:min-samples min-samples (or/c #f (integer-in 1 1023)) #f]
                      [#:cluster-selection-epsilon cluster-selection-epsilon (and/c real? (>=/c 0)) 0.0]
                      [#:max-cluster-size max-cluster-size exact-nonnegative-integer? 0]
                      [#:alpha alpha (and/c real? (>/c 0)) 1.0]
                      [#:cluster-selection-method cluster-selection-method (or/c 'eom 'leaf) 'eom]
                      [#:allow-single-cluster allow-single-cluster boolean? #f]
                      [#:params params (or/c #f hash?) #f]
                      [#:resources resources device-resources? (current-device-resources)])
         hdbscan-result?]{

Clusters the rows of @racket[X], a row-major @racket['float32] matrix with at
least 2 rows and at least @racket[min-samples] of them, by the density of their
neighbourhoods. A sample's core distance is the distance to its
@racket[min-samples]-th nearest other sample (@racket[#f] is
@racket[min-cluster-size]), so this is scikit-learn's @tt{HDBSCAN} with
@tt{min_samples} one larger. Clusters smaller than @racket[min-cluster-size]
are noise; a group of at least @racket[min-cluster-size] equal rows always
forms a cluster. @racket[cluster-selection-epsilon] merges clusters closer
than it, @racket[max-cluster-size] (0 is unlimited) caps the clusters
@racket['eom] selects, @racket[alpha] scales the distances, and
@racket[allow-single-cluster] lets the root be the one cluster. Every row
must have a squared norm of at most 2@superscript{125}.

The result keeps @racket[X] by reference, for the prediction procedures:
change it and they answer for the new rows. The arrays live on
@racket[resources]. @racket[params], from @racket[hdbscan-params], gives every
hyperparameter a keyword omits; only the euclidean @racket[metric] and the
brute-force nearest neighbours of @racket[build-algo] are bound.}

@defproc[(hdbscan-approximate-predict [result hdbscan-result?]
                                      [points device-matrix?]
                                      [#:resources resources device-resources? (current-device-resources)])
         hdbscan-prediction?]{

Labels each row of @racket[points] with the cluster it would join, without
refitting. @racket[points] is a row-major @racket['float32] matrix with at
least one row and as many columns as the fit's @racket[X], whose rows have
squared norms of at most 2@superscript{125}. Prediction needs at least 2 ×
@racket[min-samples] training rows. A released model is refused.

@examples[#:eval ev
(define few (hdbscan-fit (list*->device-matrix '((0 0) (0 1) (1 0) (1 1) (9 9)) #:dtype 'float32)
                         #:min-cluster-size 2
                         #:min-samples 3))
(eval:error (hdbscan-approximate-predict few X))
]}

@defproc[(hdbscan-membership-vector [result hdbscan-result?]
                                    [points device-matrix?]
                                    [#:batch-size batch-size exact-positive-integer? 4096]
                                    [#:resources resources device-resources? (current-device-resources)])
         device-matrix?]{

Each row of @racket[points]' soft membership in every selected cluster: an
m×k @racket['float32] matrix, m×0 when the fit found no cluster.
@racket[points] is checked as for @racket[hdbscan-approximate-predict]. A row
sums to at most 1, not to 1: the rest is the chance that the point is noise.
On the training @racket[X] the rows differ from
@racket[hdbscan-all-points-membership-vectors]', which knows each row's place
in the tree. @racket[batch-size] rows are processed at a time, with the same
answer for any batch size.}

@defproc[(hdbscan-all-points-membership-vectors [result hdbscan-result?]
                                                [#:batch-size batch-size exact-positive-integer? 4096]
                                                [#:resources resources device-resources? (current-device-resources)])
         device-matrix?]{

The soft membership of every training row: an n×k @racket['float32] matrix,
n×0 when the fit found no cluster, whose rows sum to at most 1. On a clustered
row, the largest entry is the row's label.}

@defproc[(hdbscan-params [base (or/c #f hash?) #f]
                         [#:min-cluster-size min-cluster-size (and/c exact-integer? (>=/c 2)) 5]
                         [#:min-samples min-samples (or/c #f (integer-in 1 1023)) #f]
                         [#:cluster-selection-epsilon cluster-selection-epsilon (and/c real? (>=/c 0)) 0.0]
                         [#:max-cluster-size max-cluster-size exact-nonnegative-integer? 0]
                         [#:metric metric 'euclidean 'euclidean]
                         [#:alpha alpha (and/c real? (>/c 0)) 1.0]
                         [#:cluster-selection-method cluster-selection-method (or/c 'eom 'leaf) 'eom]
                         [#:allow-single-cluster allow-single-cluster boolean? #f]
                         [#:build-algo build-algo 'brute-force 'brute-force])
         (and/c hash? hash-eq? immutable?)]{

The hyperparameters of @racket[hdbscan-fit] and @racket[hdbscan], as
@racket[kmeans-params] gives k-means'.}

@(close-eval ev)
