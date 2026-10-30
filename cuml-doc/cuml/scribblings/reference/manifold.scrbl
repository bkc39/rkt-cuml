#lang scribble/manual
@(require scribble/example
          (for-label cuml
                     racket/base
                     racket/contract/base
                     raft
                     threading))

@(define ev (make-base-eval #:lang 'racket/base '(require cuml raft threading)))

@title[#:tag "manifold"]{Manifold learning}

@defmodule[cuml/manifold #:no-declare]
@declare-exporting[cuml/manifold cuml]

@section[#:tag "umap"]{UMAP}

@examples[#:eval ev
(define X
  (list*->device-matrix
   (for*/list ([c (in-list '(0 25 50))] [i (in-range 20)])
     (list (+ c (* 0.1 i)) (- c (* 0.05 i)) (* 0.01 i i)))
   #:dtype 'float32))
(define truth (list->device-vector (for*/list ([k 3] [_ 20]) k) #:dtype 'int32))
(define fitted (umap-fit X #:n-neighbors 10))
(shape (umap-result-embedding fitted))
(~> fitted
    umap-result-embedding
    (hdbscan-fit #:min-cluster-size 10)
    hdbscan-result-labels
    (adjusted-rand-score truth _))
]

@defstruct*[umap-result
            ([embedding device-matrix?]
             [X device-matrix?]
             [a flonum?]
             [b flonum?]
             [params hash?])]{
The result of @racket[umap-fit]: the n×k @racket['float32] @racket[embedding];
the training matrix @racket[X], kept by reference for @racket[umap-transform],
so writing to it afterwards changes later transforms; the curve parameters
@racket[a] and @racket[b] the optimisation used, in float32; and the fit's
@racket[params].}

@defproc[(umap-fit [X device-matrix?]
                   [#:y y (or/c #f device-vector?) #f]
                   [#:n-neighbors n-neighbors (integer-in 2 2147483647) 15]
                   [#:n-components n-components (integer-in 1 2147483647) 2]
                   [#:metric metric (or/c 'euclidean 'sqeuclidean 'manhattan 'chebyshev 'cosine) 'euclidean]
                   [#:n-epochs n-epochs (or/c #f (integer-in 1 2147483647)) #f]
                   [#:learning-rate learning-rate (and/c real? positive?) 1.0]
                   [#:min-dist min-dist (and/c real? (>=/c 0)) 0.1]
                   [#:spread spread (and/c real? positive?) 1.0]
                   [#:a a (or/c #f (and/c real? positive?)) #f]
                   [#:b b (or/c #f (and/c real? positive?)) #f]
                   [#:init init (or/c 'spectral 'random) 'spectral]
                   [#:negative-sample-rate negative-sample-rate (integer-in 0 2147483647) 5]
                   [#:random-state random-state (or/c #f (integer-in 0 18446744073709551615)) 0]
                   [#:build-algo build-algo (or/c 'brute-force 'nn-descent) 'brute-force]
                   [#:params params (or/c #f hash?) #f]
                   [#:resources resources device-resources? (current-device-resources)])
         umap-result?]{
Embeds the rows of @racket[X], a row-major @racket['float32] matrix with at
least two rows and finite rows of squared norm at most 2@superscript{125},
in @racket[n-components] dimensions. Under @racket['cosine], each row of
@racket[X]'s d columns also needs a squared norm of at least
d·2@superscript{-126}, so a zero row is refused. Neighbourhoods use
@racket[n-neighbors] neighbours, at most the row count. @racket[y],
@racket['int32] or @racket['int64] labels with @racket[-1] for unlabelled,
supervises the embedding. @racket[a] and @racket[b], given together, each
between 2@superscript{-126} and the largest float32, set the curve that
@racket[min-dist] and @racket[spread] otherwise fit; @racket[min-dist] is
at most @racket[spread]. @racket[learning-rate] is at least
2@superscript{-126} once rounded to float32. @racket[n-epochs] defaults to
500, or 200 above 10000 rows; a fit with @racket[n-epochs] 1 or 2 cannot be
transformed. @racket['spectral] needs at least
2(@racket[n-components] + 2) rows, below which cuML's eigensolver reads and
writes past its basis or crashes. Where the spectral solve fails, as on 12
identical rows with labels i mod 3, one component and @racket[random-state]
1, the fit starts from a random embedding instead; @racket[params] still
say @racket['spectral]. On a degenerate supervised graph the spectral solve can
instead return NaN, which is refused; use @racket['random] there.
@racket['nn-descent] needs 150 rows, supports only @racket['euclidean],
@racket['sqeuclidean] and @racket['cosine], and is not
reproducible. A @racket[random-state] gives the same embedding on every run;
@racket[#f] draws a seed from @racket[current-pseudo-random-generator].
@examples[#:eval ev #:label #f
(umap-result-a (umap-fit X #:n-neighbors 10))
(eval:error (umap-fit X #:min-dist 2.0))
]}

@defproc[(umap-transform [result umap-result?]
                         [X device-matrix?]
                         [#:resources resources device-resources? (current-device-resources)])
         device-matrix?]{
Places new rows of @racket[X] into @racket[result]'s embedding, with the
fit's parameters: each starts at the weighted mean of its neighbours among
the training rows and is optimised for a third of @racket[n-epochs] (100,
or 30 above 10000 rows, when unset), at a quarter of the learning rate. The
training rows themselves come out near, not at, their embedding. Both
@racket[X] and @racket[result]'s training matrix are checked as
@racket[umap-fit] checks its @racket[X]. Where neighbour distances tie, as
among the integer rows of scikit-learn's digits, two transforms of the same
rows can differ, as they do in Python cuML.
@examples[#:eval ev #:label #f
(shape (umap-transform fitted X))
]}

@defproc[(umap-fit-transform [X device-matrix?]
                             [#:y y (or/c #f device-vector?) #f]
                             [#:n-neighbors n-neighbors (integer-in 2 2147483647) 15]
                             [#:n-components n-components (integer-in 1 2147483647) 2]
                             [#:metric metric (or/c 'euclidean 'sqeuclidean 'manhattan 'chebyshev 'cosine) 'euclidean]
                             [#:n-epochs n-epochs (or/c #f (integer-in 1 2147483647)) #f]
                             [#:learning-rate learning-rate (and/c real? positive?) 1.0]
                             [#:min-dist min-dist (and/c real? (>=/c 0)) 0.1]
                             [#:spread spread (and/c real? positive?) 1.0]
                             [#:a a (or/c #f (and/c real? positive?)) #f]
                             [#:b b (or/c #f (and/c real? positive?)) #f]
                             [#:init init (or/c 'spectral 'random) 'spectral]
                             [#:negative-sample-rate negative-sample-rate (integer-in 0 2147483647) 5]
                             [#:random-state random-state (or/c #f (integer-in 0 18446744073709551615)) 0]
                             [#:build-algo build-algo (or/c 'brute-force 'nn-descent) 'brute-force]
                             [#:params params (or/c #f hash?) #f]
                             [#:resources resources device-resources? (current-device-resources)])
         (values umap-result? device-matrix?)]{
@racket[umap-fit], and the fit's own embedding.}

@defproc[(umap-params [base (or/c #f hash?) #f]
                      [#:n-neighbors n-neighbors (integer-in 2 2147483647) 15]
                      [#:n-components n-components (integer-in 1 2147483647) 2]
                      [#:metric metric (or/c 'euclidean 'sqeuclidean 'manhattan 'chebyshev 'cosine) 'euclidean]
                      [#:n-epochs n-epochs (or/c #f (integer-in 1 2147483647)) #f]
                      [#:learning-rate learning-rate (and/c real? positive?) 1.0]
                      [#:min-dist min-dist (and/c real? (>=/c 0)) 0.1]
                      [#:spread spread (and/c real? positive?) 1.0]
                      [#:a a (or/c #f (and/c real? positive?)) #f]
                      [#:b b (or/c #f (and/c real? positive?)) #f]
                      [#:init init (or/c 'spectral 'random) 'spectral]
                      [#:negative-sample-rate negative-sample-rate (integer-in 0 2147483647) 5]
                      [#:random-state random-state (or/c #f (integer-in 0 18446744073709551615)) 0]
                      [#:build-algo build-algo (or/c 'brute-force 'nn-descent) 'brute-force])
         (and/c hash? hash-eq? immutable?)]{
The hyperparameters of @racket[umap-fit] and @racket[umap], as @racket[kmeans-params].
@examples[#:eval ev #:label #f
(hash-ref (umap-params #:init 'random) 'init)
(eval:error (umap-params (hasheq 'local-connectivity 2.0)))
]}

@section[#:tag "umap-estimator"]{UMAP estimator}

@declare-exporting[cuml]

@defproc[(umap [params (or/c #f hash?) #f]
               [#:n-neighbors n-neighbors (integer-in 2 2147483647) 15]
               [#:n-components n-components (integer-in 1 2147483647) 2]
               [#:metric metric (or/c 'euclidean 'sqeuclidean 'manhattan 'chebyshev 'cosine) 'euclidean]
               [#:n-epochs n-epochs (or/c #f (integer-in 1 2147483647)) #f]
               [#:learning-rate learning-rate (and/c real? positive?) 1.0]
               [#:min-dist min-dist (and/c real? (>=/c 0)) 0.1]
               [#:spread spread (and/c real? positive?) 1.0]
               [#:a a (or/c #f (and/c real? positive?)) #f]
               [#:b b (or/c #f (and/c real? positive?)) #f]
               [#:init init (or/c 'spectral 'random) 'spectral]
               [#:negative-sample-rate negative-sample-rate (integer-in 0 2147483647) 5]
               [#:random-state random-state (or/c #f (integer-in 0 18446744073709551615)) 0]
               [#:build-algo build-algo (or/c 'brute-force 'nn-descent) 'brute-force]
               [#:resources resources (or/c #f device-resources?) #f])
         umap?]{

An unfitted UMAP with the hyperparameters @racket[(umap-params params ...)]:
an estimator and a transformer. @racket[fit] is @racket[umap-fit], with
@racket[y] as its @racket[#:y]; @racket[transform] is @racket[umap-transform];
@racket[fit-transform] returns the fit's embedding. A @racket[sample-weight]
is refused. Errors name the generic.

@examples[#:eval ev #:label #f
(shape (fit-transform (umap #:n-neighbors 10) X))
(eval:error (transform (umap) X))
]}

@defproc[(umap? [v any/c]) boolean?]{

Whether @racket[v] is a UMAP estimator.}

@defproc[(umap-model [estimator umap?]) (or/c #f umap-result?)]{

The @racket[umap-result] of the fit that made @racket[estimator], or
@racket[#f] when it is unfitted.}

@section[#:tag "tsne"]{t-SNE}

@examples[#:eval ev
(define embedded (tsne-fit X #:perplexity 5))
(shape (tsne-result-embedding embedded))
(~> embedded
    tsne-result-embedding
    (hdbscan-fit #:min-cluster-size 10)
    hdbscan-result-labels
    (adjusted-rand-score truth _))
]

@defstruct*[tsne-result
            ([embedding device-matrix?]
             [kl-divergence (or/c #f flonum?)]
             [n-iter exact-positive-integer?]
             [params hash?])]{
The result of @racket[tsne-fit]: the n×2 @racket['float32] @racket[embedding];
the Kullback–Leibler divergence of the last iteration, in float32, or
@racket[#f] when the optimisation stopped before @racket[max-iter]; the
iterations run; and the fit's @racket[params], where a derived
@racket[n-neighbors] and a drawn seed stay @racket[#f].}

@defproc[(tsne-fit [X device-matrix?]
                   [#:n-components n-components 2 2]
                   [#:perplexity perplexity (and/c real? (>/c 1)) 30.0]
                   [#:early-exaggeration early-exaggeration (and/c real? (>=/c 1)) 12.0]
                   [#:learning-rate learning-rate (and/c real? positive?) 200.0]
                   [#:max-iter max-iter (integer-in 1 2147483647) 1000]
                   [#:n-neighbors n-neighbors (or/c #f (integer-in 2 1023)) #f]
                   [#:method method (or/c 'fft 'exact) 'fft]
                   [#:metric metric (or/c 'euclidean 'sqeuclidean 'manhattan 'chebyshev 'cosine) 'euclidean]
                   [#:init init (or/c 'random 'pca) 'random]
                   [#:random-state random-state (or/c #f (integer-in 0 9223372036854775807)) 0]
                   [#:params params (or/c #f hash?) #f]
                   [#:resources resources device-resources? (current-device-resources)])
         tsne-result?]{
Embeds the rows of @racket[X], a row-major @racket['float32] matrix with at
least two rows and two columns, not all rows equal, and finite rows of squared
norm at most 2@superscript{125}, in two dimensions. Under @racket['cosine] a
row of d columns needs a squared norm of at least d·2@superscript{-126},
under @racket['sqeuclidean] one of at most 2@superscript{61}, and under
@racket['manhattan] an L1 norm of at most 2@superscript{62.5}. Under
@racket['pca] init the rows' squared distances from their mean sum to at most
2@superscript{126}. A fit whose largest neighbour distance under the metric
is below 2@superscript{-64} (about 5·10@superscript{-20}) diverges. Under
@racket['euclidean] and @racket['sqeuclidean] the distances are
‖x‖²+‖y‖²−2x·y in @racket['float32]: centre @racket[X] if it lies far from
the origin. Duplicate rows that fill every row's neighbourhood diverge: raise
@racket[n-neighbors], or deduplicate @racket[X] once every row has 1023 or
more copies.
Each row's neighbourhood is its @racket[n-neighbors] nearest rows, itself
included: ⌈3·@racket[perplexity]⌉ by default, at least 2, at most 1023 and at
most the row count; @racket[perplexity] is above 1 in float32 and at most that
count. The first 250
iterations run at @racket[learning-rate] with attractions scaled by
@racket[early-exaggeration], the rest at twice the rate. @racket['fft] runs
every iteration, where Python cuML's default stops early and can collapse the
embedding. @racket['fft] with a @racket[random-state] gives the same embedding
on every run; @racket['exact] is O(n²) per iteration, takes at most 2097120
rows, stops once the gradient norm falls below 10@superscript{-7}, checked
every 10th iteration after the first 250, and is not
reproducible. @racket[#f] draws a seed from
@racket[current-pseudo-random-generator]. An optimisation that diverges is an
error.
@examples[#:eval ev #:label #f
(tsne-result-n-iter embedded)
(eval:error (tsne-fit X #:perplexity 100))
]}

@defproc[(tsne-fit-transform [X device-matrix?]
                             [#:n-components n-components 2 2]
                             [#:perplexity perplexity (and/c real? (>/c 1)) 30.0]
                             [#:early-exaggeration early-exaggeration (and/c real? (>=/c 1)) 12.0]
                             [#:learning-rate learning-rate (and/c real? positive?) 200.0]
                             [#:max-iter max-iter (integer-in 1 2147483647) 1000]
                             [#:n-neighbors n-neighbors (or/c #f (integer-in 2 1023)) #f]
                             [#:method method (or/c 'fft 'exact) 'fft]
                             [#:metric metric (or/c 'euclidean 'sqeuclidean 'manhattan 'chebyshev 'cosine) 'euclidean]
                             [#:init init (or/c 'random 'pca) 'random]
                             [#:random-state random-state (or/c #f (integer-in 0 9223372036854775807)) 0]
                             [#:params params (or/c #f hash?) #f]
                             [#:resources resources device-resources? (current-device-resources)])
         (values tsne-result? device-matrix?)]{
@racket[tsne-fit], and the fit's embedding.}

@defproc[(tsne-params [base (or/c #f hash?) #f]
                      [#:n-components n-components 2 2]
                      [#:perplexity perplexity (and/c real? (>/c 1)) 30.0]
                      [#:early-exaggeration early-exaggeration (and/c real? (>=/c 1)) 12.0]
                      [#:learning-rate learning-rate (and/c real? positive?) 200.0]
                      [#:max-iter max-iter (integer-in 1 2147483647) 1000]
                      [#:n-neighbors n-neighbors (or/c #f (integer-in 2 1023)) #f]
                      [#:method method (or/c 'fft 'exact) 'fft]
                      [#:metric metric (or/c 'euclidean 'sqeuclidean 'manhattan 'chebyshev 'cosine) 'euclidean]
                      [#:init init (or/c 'random 'pca) 'random]
                      [#:random-state random-state (or/c #f (integer-in 0 9223372036854775807)) 0])
         (and/c hash? hash-eq? immutable?)]{
The hyperparameters of @racket[tsne-fit] and @racket[tsne], as @racket[kmeans-params].
@examples[#:eval ev #:label #f
(hash-ref (tsne-params #:method 'exact) 'method)
(eval:error (tsne-params (hasheq 'angle 0.5)))
]}

@section[#:tag "tsne-estimator"]{t-SNE estimator}

@declare-exporting[cuml]

@defproc[(tsne [params (or/c #f hash?) #f]
               [#:n-components n-components 2 2]
               [#:perplexity perplexity (and/c real? (>/c 1)) 30.0]
               [#:early-exaggeration early-exaggeration (and/c real? (>=/c 1)) 12.0]
               [#:learning-rate learning-rate (and/c real? positive?) 200.0]
               [#:max-iter max-iter (integer-in 1 2147483647) 1000]
               [#:n-neighbors n-neighbors (or/c #f (integer-in 2 1023)) #f]
               [#:method method (or/c 'fft 'exact) 'fft]
               [#:metric metric (or/c 'euclidean 'sqeuclidean 'manhattan 'chebyshev 'cosine) 'euclidean]
               [#:init init (or/c 'random 'pca) 'random]
               [#:random-state random-state (or/c #f (integer-in 0 9223372036854775807)) 0]
               [#:resources resources (or/c #f device-resources?) #f])
         tsne?]{

An unfitted t-SNE with the hyperparameters @racket[(tsne-params params ...)]:
an estimator and a transformer whose @racket[fit-transform] returns the
embedding. @racket[fit] is @racket[tsne-fit]; there is no @racket[transform].
A @racket[sample-weight] is refused. Errors name the generic.

@examples[#:eval ev #:label #f
(shape (fit-transform (tsne #:perplexity 5) X))
(eval:error (transform (fit (tsne #:perplexity 5) X) X))
]}

@defproc[(tsne? [v any/c]) boolean?]{

Whether @racket[v] is a t-SNE estimator.}

@defproc[(tsne-model [estimator tsne?]) (or/c #f tsne-result?)]{

The @racket[tsne-result] of the fit that made @racket[estimator], or
@racket[#f] when it is unfitted.}

@section[#:tag "spectral-embedding"]{Spectral embedding}

@examples[#:eval ev
(define spectral (spectral-embedding-fit X #:n-neighbors 10))
(shape (spectral-embedding-result-embedding spectral))
(~> spectral
    spectral-embedding-result-embedding
    (kmeans-fit #:n-clusters 3)
    kmeans-result-labels
    (adjusted-rand-score truth _))
]

@defstruct*[spectral-embedding-result
            ([embedding device-matrix?]
             [n-neighbors (or/c #f exact-positive-integer?)]
             [params hash?])]{
The result of @racket[spectral-embedding-fit] and
@racket[spectral-embedding-fit-graph]: the n×k @racket['float32]
@racket[embedding]; the neighbour count the graph used, or @racket[#f] for a
given graph; and the fit's @racket[params], where a derived
@racket[n-neighbors] and a drawn seed stay @racket[#f].}

@defproc[(spectral-embedding-fit [X device-matrix?]
                                 [#:n-components n-components exact-positive-integer? 2]
                                 [#:n-neighbors n-neighbors (or/c #f (integer-in 2 2147483647)) #f]
                                 [#:norm-laplacian norm-laplacian boolean? #t]
                                 [#:drop-first drop-first boolean? #t]
                                 [#:random-state random-state (or/c #f (integer-in 0 18446744073709551615)) 0]
                                 [#:params params (or/c #f hash?) #f]
                                 [#:resources resources device-resources? (current-device-resources)])
         spectral-embedding-result?]{
Embeds the rows of @racket[X], a row-major @racket['float32] matrix, in
@racket[n-components] dimensions: the eigenvectors of the smallest eigenvalues
of the graph Laplacian, normalized when @racket[norm-laplacian] and then
divided by the square root of each degree. The constant first one is dropped
when @racket[drop-first].
The graph joins each row to its @racket[n-neighbors] nearest rows, itself
included, with weight 1, made symmetric. @racket[n-neighbors] defaults to
min(256, max(2, ⌊n/10⌋)) and is at most the row count.
@racket[X] needs at least two columns, two distinct rows, and finite rows of
squared norm at most 2@superscript{125}. It needs 2(@racket[n-components] + 2)
rows with @racket[drop-first], 2(@racket[n-components] + 1) without.
Each column's largest entry is positive. A @racket[random-state] repeats
bitwise when @racket[n-neighbors] is at most 256; @racket[#f] draws one from
@racket[current-pseudo-random-generator].
Eigenvectors of close eigenvalues are approximate.
A graph whose Laplacian repeats eigenvalues (rings, grids, complete graphs,
stars, identical pieces) can give a wrong embedding or an error, with either
Laplacian.
This procedure cannot check its eigenvectors;
@racket[spectral-embedding-fit-graph] refuses the inaccurate ones it detects.
Centre @racket[X] if it lies far from the origin.
An eigensolver result with a NaN or a zero column, such as from duplicated
rows, is an error.
@examples[#:eval ev #:label #f
(spectral-embedding-result-n-neighbors spectral)
(eval:error (spectral-embedding-fit X #:n-components 30))
]}

@defproc[(spectral-embedding-fit-graph [rows device-vector?]
                                       [cols device-vector?]
                                       [vals device-vector?]
                                       [n-samples (integer-in 1 2147483647)]
                                       [#:n-components n-components exact-positive-integer? 2]
                                       [#:norm-laplacian norm-laplacian boolean? #t]
                                       [#:drop-first drop-first boolean? #t]
                                       [#:random-state random-state (or/c #f (integer-in 0 18446744073709551615)) 0]
                                       [#:params params (or/c #f hash?) #f]
                                       [#:resources resources device-resources? (current-device-resources)])
         spectral-embedding-result?]{
Embeds @racket[n-samples] samples from an affinity graph given as
coordinates, as @racket[spectral-clustering-fit-graph] takes it: symmetric,
without self-loops, each entry 0 or at least the smallest normal
@racket['float32], duplicates then adding. A graph
that is not symmetric is used as given: an inaccurate or NaN eigenvector is
refused, the error saying the graph is not symmetric, and cuML's eigensolver
can fail to converge or return a zero column.
Without @racket[norm-laplacian] the embedding does not depend on the scale of
the weights; with it, it scales by 1/√c when the weights scale by c.
Without @racket[norm-laplacian], weights spanning orders of magnitude can give
a wrong embedding or an error; use @racket[norm-laplacian].
A column whose eigen-residual passes 10⁻³ of the largest degree (of the
normalised Laplacian with @racket[norm-laplacian]) is an error, as is a
constant one with @racket[drop-first].
With @racket[norm-laplacian] a sample without edges has eigenvalue 1, and many
such samples can displace the graph's eigenvectors.
Without @racket[norm-laplacian] a sample without edges has eigenvalue 0, as
any separate piece does, and takes a leading column.
Either such fit can be returned without an error. Otherwise as
@racket[spectral-embedding-fit], except that @racket[params]'s
@racket[n-neighbors] and @racket[affinity] are ignored.
@examples[#:eval ev #:label #f
(define path-n 24)
(define path (for*/list ([i (sub1 path-n)] [e (list (cons i (add1 i)) (cons (add1 i) i))]) e))
(define path-rows (list->device-vector (map car path) #:dtype 'int32))
(define path-cols (list->device-vector (map cdr path) #:dtype 'int32))
(define path-vals (list->device-vector (map (lambda (_) 1.0) path) #:dtype 'float32))
(~> (spectral-embedding-fit-graph path-rows path-cols path-vals path-n #:n-components 1)
    spectral-embedding-result-embedding
    shape)
(eval:error (spectral-embedding-fit-graph path-rows path-rows path-vals path-n))
]}

@defproc[(spectral-embedding-fit-transform [X device-matrix?]
                                           [#:n-components n-components exact-positive-integer? 2]
                                           [#:n-neighbors n-neighbors (or/c #f (integer-in 2 2147483647)) #f]
                                           [#:norm-laplacian norm-laplacian boolean? #t]
                                           [#:drop-first drop-first boolean? #t]
                                           [#:random-state random-state (or/c #f (integer-in 0 18446744073709551615)) 0]
                                           [#:params params (or/c #f hash?) #f]
                                           [#:resources resources device-resources? (current-device-resources)])
         (values spectral-embedding-result? device-matrix?)]{
@racket[spectral-embedding-fit], and the fit's embedding.}

@defproc[(spectral-embedding-params [base (or/c #f hash?) #f]
                                    [#:n-components n-components exact-positive-integer? 2]
                                    [#:n-neighbors n-neighbors (or/c #f (integer-in 2 2147483647)) #f]
                                    [#:norm-laplacian norm-laplacian boolean? #t]
                                    [#:drop-first drop-first boolean? #t]
                                    [#:random-state random-state (or/c #f (integer-in 0 18446744073709551615)) 0]
                                    [#:affinity affinity 'nearest-neighbors 'nearest-neighbors])
         (and/c hash? hash-eq? immutable?)]{
The hyperparameters of @racket[spectral-embedding-fit],
@racket[spectral-embedding-fit-graph] and @racket[spectral-embedding], as
@racket[kmeans-params] gives k-means'. @racket[affinity] is cuML's name for
the graph built from @racket[X]; only @racket['nearest-neighbors] fits.
@examples[#:eval ev #:label #f
(hash-ref (spectral-embedding-params #:drop-first #f) 'drop-first)
(eval:error (spectral-embedding-fit X #:params (spectral-embedding-params #:affinity 'precomputed)))
]}

@section[#:tag "spectral-embedding-estimator"]{Spectral embedding estimator}

@declare-exporting[cuml]

@defproc[(spectral-embedding [params (or/c #f hash?) #f]
                             [#:n-components n-components exact-positive-integer? 2]
                             [#:n-neighbors n-neighbors (or/c #f (integer-in 2 2147483647)) #f]
                             [#:norm-laplacian norm-laplacian boolean? #t]
                             [#:drop-first drop-first boolean? #t]
                             [#:random-state random-state (or/c #f (integer-in 0 18446744073709551615)) 0]
                             [#:affinity affinity 'nearest-neighbors 'nearest-neighbors]
                             [#:resources resources (or/c #f device-resources?) #f])
         spectral-embedding?]{

An unfitted spectral embedding with the hyperparameters
@racket[(spectral-embedding-params params ...)]: an estimator and a
transformer whose @racket[fit-transform] returns the embedding. @racket[fit]
is @racket[spectral-embedding-fit]; there is no @racket[transform]. A
@racket[sample-weight] is refused. Errors name the generic.

@examples[#:eval ev #:label #f
(shape (fit-transform (spectral-embedding #:n-neighbors 10) X))
(eval:error (transform (fit (spectral-embedding #:n-neighbors 10) X) X))
]}

@defproc[(spectral-embedding? [v any/c]) boolean?]{

Whether @racket[v] is a spectral embedding estimator.}

@defproc[(spectral-embedding-model [estimator spectral-embedding?])
         (or/c #f spectral-embedding-result?)]{

The @racket[spectral-embedding-result] of the fit that made
@racket[estimator], or @racket[#f] when it is unfitted.}

@(close-eval ev)
