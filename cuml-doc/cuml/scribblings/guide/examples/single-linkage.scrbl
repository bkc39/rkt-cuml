#lang scribble/manual
@(require "../../utils.rkt")

@(define ev (make-cuml-eval))

@title[#:tag "ex-single-linkage"]{Single linkage: iris and chaining}

Single linkage starts from one cluster per sample and repeatedly merges the
two clusters with the closest pair of samples. It follows gaps, not centres,
so it finds clusters of any shape that a gap separates, and it chains: two
groups joined by a run of close samples merge into one.

@section[#:tag "ex-single-linkage-data"]{The data}

The iris measurements of @secref["getting-started"]:

@examples[#:eval ev #:label #f
(define flowers (dataset-matrix 'iris (features 'iris "species")))
(define species (dataset-column 'iris "species"))
]

@section[#:tag "ex-single-linkage-two"]{Two clusters}

@examples[#:eval ev #:label #f
(define (single-linkage-table k)
  (crosstab (labels-of (agglomerative-clustering #:n-clusters k) flowers) species))
(single-linkage-table 2)
]

A clear gap separates setosa from the other two species, and single linkage
finds it exactly.

@section[#:tag "ex-single-linkage-chaining"]{Chaining}

Versicolor and virginica touch. Asking for three clusters does not split them;
it cuts off two virginica flowers that lie apart from the rest:

@examples[#:eval ev #:label #f
(single-linkage-table 3)
(agreement (single-linkage-table 3))
]

Only 102 of 150 flowers are in a cluster whose majority is their species,
against 134 for k-means in @secref["gs-species"]. More clusters peel off more
outliers, and at six the touching pair is still one cluster:

@examples[#:eval ev #:label #f
(single-linkage-table 6)
]

Use single linkage when the clusters are separated by gaps, whatever their
shape, and expect outliers as clusters of their own; for touching clusters
use k-means or spectral clustering.

@(close-eval ev)
