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
(define flowers
  (table->device-matrix (load-dataset-table 'iris) #:columns (features 'iris "species")))
(define species (dataset-column 'iris "species"))
(define species-ids (list->device-vector (renumbered species) #:dtype 'int32))
(define (counts-of labels)
  (~> (contingency-matrix species-ids labels) contingency-table-counts device-matrix->list*))
]

@section[#:tag "ex-single-linkage-two"]{Two clusters}

@examples[#:eval ev #:label #f
(define (single-linkage-labels k)
  (fit-predict (agglomerative-clustering #:n-clusters k) flowers))
(counts-of (single-linkage-labels 2))
]

A clear gap separates setosa from the other two species, and single linkage
finds it exactly.

@section[#:tag "ex-single-linkage-chaining"]{Chaining}

Versicolor and virginica touch. Asking for three clusters does not split them;
it cuts off two virginica flowers that lie apart from the rest:

@examples[#:eval ev #:label #f
(counts-of (single-linkage-labels 3))
(~r (adjusted-rand-score species-ids (single-linkage-labels 3)) #:precision 3)
]

Its adjusted Rand index is 0.564, against 0.73 for k-means in
@secref["gs-species"]. More clusters peel off more
outliers, and at six the touching pair is still one cluster:

@examples[#:eval ev #:label #f
(counts-of (single-linkage-labels 6))
]

The cuts by petal length and width. A cluster takes the colour of the species
it shares the most flowers with; the clusters left over take colours of their
own:

@examples[#:eval ev #:label #f
(define petal-length (dataset-column 'iris "petal-length"))
(define petal-width (dataset-column 'iris "petal-width"))
(define cluster-counts '(2 3 4 6))
(define cuts
  (for/list ([k (in-list cluster-counts)])
    (clusters-of (agglomerative-clustering #:n-clusters k) flowers species)))
(sweep-figure petal-length petal-width
              (map (curry format "~a clusters") cluster-counts)
              cuts
              #:names (remove-duplicates species)
              #:x-label "petal length (cm)"
              #:y-label "petal width (cm)")
]

At every cut, 92 or more of the 100 versicolor and virginica flowers stay one
orange cluster; each further cluster is a few flowers at its edge.

Use single linkage when the clusters are separated by gaps, whatever their
shape, and expect outliers as clusters of their own; for touching clusters
use k-means or spectral clustering.

@(close-eval ev)
