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

@racket[r2-score] and @racket[mean-squared-error] take @racket['float32] or
@racket['float64] values, either one in each argument. Each score takes an
optional @racket[sample-weight], a @racket['float32] or @racket['float64]
vector of finite values of at least 0 with a positive sum, and sums in
double precision. A weight of at most 2@superscript{−1075}·2@superscript{e},
where 2@superscript{e} is the largest weight rounded down to a power of two,
counts as zero: its sample is dropped. A weight w below
2@superscript{−1022}·2@superscript{e} is held as a multiple of
2@superscript{−1074}·2@superscript{e}, a relative error of up to
2@superscript{e−1075}/w, and the terms of its sample are rounded on the
same grid, so a score that such samples carry is accurate only to about
that. Repeated calls on the same data give the same result.

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
whose squared pairwise distances (plain distances, under @racket['manhattan]
and @racket['chebyshev]) come near m lose precision; rescale it first.
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

@defproc[(r2-score [y-true device-vector?]
                   [y-pred device-vector?]
                   [#:sample-weight sample-weight (or/c #f device-vector?) #f]
                   [#:resources resources device-resources? (current-device-resources)])
         flonum?]{
The coefficient of determination of @racket[y-pred] against @racket[y-true],
vectors of one length: 1 − Σw(y − ŷ)²/Σw(y − ȳ)², with ȳ the weighted mean
of @racket[y-true] and w the weights (all 1 when @racket[sample-weight] is
@racket[#f]). 1.0 is a perfect fit, 0.0 the constant ȳ, and worse fits are
negative. When @racket[y-true] is constant over the samples of positive
weight, as with one sample, the score is 1.0 if @racket[y-pred] equals it
there and 0.0 otherwise; sklearn returns NaN for one sample and decides
constancy from the rounded variance. Every value must be finite.
@examples[#:eval ev
(define observed (list->device-vector '(3 -0.5 2 7) #:dtype 'float64))
(define fitted (list->device-vector '(2.5 0.0 2 8) #:dtype 'float64))
(r2-score observed fitted)
(r2-score observed observed)
(r2-score (list->device-vector '(2 2 2) #:dtype 'float32)
          (list->device-vector '(2 2 3) #:dtype 'float32))]}

@defproc[(mean-squared-error [y-true (or/c device-vector? device-matrix?)]
                             [y-pred (or/c device-vector? device-matrix?)]
                             [#:sample-weight sample-weight (or/c #f device-vector?) #f]
                             [#:resources resources device-resources? (current-device-resources)])
         flonum?]{
The weighted mean of (y − ŷ)² over the samples: vectors of one length, or
row-major matrices of one shape with one row per sample, whose error is the
mean over the columns. Every value must be finite. The root mean squared
error is its @racket[sqrt].
@examples[#:eval ev
(mean-squared-error observed fitted)
(mean-squared-error (list*->device-matrix '((0.5 1) (-1 1) (7 -6)) #:dtype 'float64)
                    (list*->device-matrix '((0 2) (-1 2) (8 -5)) #:dtype 'float64))
(sqrt (mean-squared-error observed fitted))]}

@defproc[(accuracy-score [y-true device-vector?]
                         [y-pred device-vector?]
                         [#:sample-weight sample-weight (or/c #f device-vector?) #f]
                         [#:resources resources device-resources? (current-device-resources)])
         flonum?]{
The weighted fraction of samples whose label in @racket[y-pred] equals
the one in @racket[y-true]: @racket['int32] or @racket['int64] vectors of
one length, compared by value. Without weights it is the number of
matches over n, correctly rounded.
@examples[#:eval ev
(define actual (list->device-vector '(0 1 2 3) #:dtype 'int32))
(define guess (list->device-vector '(0 2 1 3) #:dtype 'int64))
(accuracy-score actual guess)
(accuracy-score actual guess
                #:sample-weight (list->device-vector '(1 1 1 5) #:dtype 'float64))]}
