#lang scribble/manual
@(require scribble/example
          (for-label cuml
                     racket/base
                     racket/contract/base
                     racket/flonum
                     racket/match
                     raft))

@(define ev (make-base-eval #:lang 'racket/base '(require cuml raft racket/match)))

@title[#:tag "metrics"]{Metrics}

@defmodule[cuml/metrics #:no-declare]
@declare-exporting[cuml/metrics cuml]

Each procedure compares labels or scores samples on the GPU, on the same
@racket[device-resources] as its arguments. Labels may be @racket['int32]
or @racket['int64] vectors, either one in each argument; only their values
matter, so @racket[-1] (noise) is a label like any other.
@racket[contingency-matrix] and @racket[adjusted-rand-score] take at most
46340 distinct values in each argument and use up to about 16·K@superscript{2}
bytes of device memory for the larger count K.

@defstruct*[contingency-table ([counts device-matrix?]
                               [row-labels device-vector?]
                               [column-labels device-vector?])]{
The result of @racket[contingency-matrix]: an @racket['int64] matrix whose
entry (i, j) counts the samples with the i-th value of @racket[row-labels]
in @racket[labels-true] and the j-th value of @racket[column-labels] in
@racket[labels-pred]. Each label vector lists the distinct values of its
argument in ascending order, in that argument's dtype.}

@defproc[(contingency-matrix [labels-true device-vector?]
                             [labels-pred device-vector?]
                             [#:resources resources device-resources? (current-device-resources)])
         contingency-table?]{
Counts how the samples of each label of @racket[labels-true] fall among
the labels of @racket[labels-pred]. Empty labels give an empty table.
@examples[#:eval ev
(define truth (list->device-vector '(0 0 0 1 1 1) #:dtype 'int32))
(define found (list->device-vector '(1 1 -1 0 0 0) #:dtype 'int64))
(match-define (contingency-table counts rows columns) (contingency-matrix truth found))
(list (device-vector->list rows) (device-vector->list columns))
(device-matrix->list* counts)]}

@defproc[(adjusted-rand-score [labels-true device-vector?]
                              [labels-pred device-vector?]
                              [#:resources resources device-resources? (current-device-resources)])
         flonum?]{
The Rand index of the two labelings corrected for chance: 1.0 when they
partition the samples identically, whatever the label values, near 0.0
for independent labelings, and negative below chance. It is symmetric.
Fewer than two samples score 1.0.
@examples[#:eval ev (adjusted-rand-score truth found) (adjusted-rand-score truth truth)]}

@defproc[(silhouette-score [X device-matrix?]
                           [labels device-vector?]
                           [#:metric metric
                                     (or/c 'euclidean 'sqeuclidean 'manhattan 'cosine 'chebyshev)
                                     'euclidean]
                           [#:chunk-size chunk-size exact-positive-integer? 40000]
                           [#:resources resources device-resources? (current-device-resources)])
         flonum?]{
The mean silhouette coefficient of the rows of @racket[X], a row-major
@racket['float32] or @racket['float64] matrix, under @racket[labels]:
for each sample, (b − a)/max(a, b), with a its mean distance to its own
cluster and b to the nearest other one; a sample alone in its cluster
scores 0. @racket[labels] must hold from 2 to n − 1 distinct values.
Distances are computed in tiles of @racket[chunk-size] rows, at most 46340;
a smaller value uses less memory. Results can differ in the last bits from
call to call. Each row's squared norm, summed in @racket[X]'s dtype, must
be at most 2@superscript{125} (@racket['float32]) or 2@superscript{1021}
(@racket['float64]), and under @racket['cosine] at least d·m for d
columns, where m is the least normal number, 2@superscript{−126} or
2@superscript{−1022}. Distances are computed in @racket[X]'s dtype, so data
whose pairwise distances (squared, under @racket['sqeuclidean]) come near m
lose precision; rescale it first.
@examples[#:eval ev
(define X (list*->device-matrix '((0 0) (0 1) (1 0) (10 10) (10 11) (11 10)) #:dtype 'float64))
(define y (list->device-vector '(0 0 0 1 1 1) #:dtype 'int32))
(silhouette-score X y)
(eval:error (silhouette-score X (list->device-vector '(0 0 0 0 0 0) #:dtype 'int32)))]}

@defproc[(silhouette-samples [X device-matrix?]
                             [labels device-vector?]
                             [#:metric metric
                                       (or/c 'euclidean 'sqeuclidean 'manhattan 'cosine 'chebyshev)
                                       'euclidean]
                             [#:chunk-size chunk-size exact-positive-integer? 40000]
                             [#:resources resources device-resources? (current-device-resources)])
         device-vector?]{
Each sample's silhouette coefficient, in @racket[X]'s dtype; their mean
is @racket[silhouette-score], whose bounds apply. Each lies in [−1, 1],
except that under @racket['cosine] rounding can carry one slightly past 1.
@examples[#:eval ev (device-vector->list (silhouette-samples X y))]}

@defproc[(trustworthiness [X device-matrix?]
                          [X-embedded device-matrix?]
                          [#:n-neighbors n-neighbors exact-positive-integer? 5]
                          [#:metric metric 'euclidean 'euclidean]
                          [#:batch-size batch-size exact-positive-integer? 512]
                          [#:resources resources device-resources? (current-device-resources)])
         flonum?]{
How well @racket[X-embedded], a row-major @racket['float32] matrix with
one row per row of @racket[X], keeps the euclidean neighbourhoods of
@racket[X]: 1.0 when each sample's @racket[n-neighbors] nearest neighbours
in the embedding are among its nearest in @racket[X], lower as the
embedding brings in samples that were far. @racket[n-neighbors] must be
less than half the rows. @racket[batch-size] rows of @racket[X] are
ranked at a time; it changes memory, not the score. Every row of both
must have a squared norm of at most 2@superscript{125}, and data whose
squared pairwise distances come near 2@superscript{−126} lose precision;
rescale it first. Tied distances can score differently from other
implementations.
@examples[#:eval ev
(define P
  (list*->device-matrix '((0 0 0) (0 1 3) (1 0 1) (3 2 0) (5 5 4) (5 6 1) (6 5 2) (7 7 7))
                        #:dtype 'float32))
(define E
  (list*->device-matrix '((0 0) (1 2) (1 0) (2 2) (5 5) (5 6) (7 6) (7 7)) #:dtype 'float32))
(trustworthiness P P #:n-neighbors 2)
(trustworthiness P E #:n-neighbors 2)]}
