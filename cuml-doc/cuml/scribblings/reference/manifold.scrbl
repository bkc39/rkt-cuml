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
(define truth (list->device-vector (for*/list ([k 3] [i 20]) k) #:dtype 'int32))
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
in @racket[n-components] dimensions. Neighbourhoods use @racket[n-neighbors]
neighbours, at most the row count. @racket[y], @racket['int32] or
@racket['int64] labels with @racket[-1] for unlabelled, supervises the
embedding. @racket[a] and @racket[b], given together, set the curve that
@racket[min-dist] and @racket[spread] otherwise fit. @racket[n-epochs]
defaults to 500, or 200 above 10000 rows. @racket['spectral] needs more rows
than components; @racket['nn-descent] needs 150 rows and is not reproducible.
A @racket[random-state] gives the same embedding on every run; @racket[#f]
draws a seed from @racket[current-pseudo-random-generator].
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
training rows themselves come out near, not at, their embedding.
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

@(close-eval ev)
