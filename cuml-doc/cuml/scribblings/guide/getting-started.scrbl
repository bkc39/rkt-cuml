#lang scribble/manual
@(require "../utils.rkt")

@(define ev (make-cuml-eval))

@title[#:tag "getting-started" #:style 'toc]{Getting started}

This chapter copies Fisher's iris flowers to the GPU, clusters them with
k-means, reads the clusters against the species and assigns new flowers to
clusters.

@local-table-of-contents[]

@section[#:tag "gs-installing"]{Installing}

@commandline{raco pkg install --auto cuml datasets-core threading}

The packages require Racket 9.3 or later, and cuML an NVIDIA GPU.
@tt{datasets-core} supplies this guide's data and @tt{threading} its
@racket[~>]. Every example in the guide starts from this require and the
helpers in @secref["examples"]:

@page-require-block

@section[#:tag "gs-data"]{Data on the device}

The iris data (R. A. Fisher, 1936, from the UCI Machine Learning Repository,
@hyperlink["https://doi.org/10.24432/C56C76"]{doi:10.24432/C56C76}, under
CC BY 4.0) measures 150 flowers of three species, 50 each, in four columns.
@racket[load-dataset-table] gives the measurements as columns;
@racket[list*->device-matrix] copies their rows to the GPU as a
@tech{sample matrix}, one flower per row:

@examples[#:eval ev #:label #f
(define measurements '("sepal-length" "sepal-width" "petal-length" "petal-width"))
(define flowers
  (~> (load-dataset-table 'iris #:columns measurements)
      dataset-table-columns
      vector->list
      (map vector->list _)
      (apply map list _)
      (list*->device-matrix #:dtype 'float32)))
(shape flowers)
]

The matrix lives on the calling thread's @racket[(current-device-resources)],
as do the results of every call below. The species stay on the host:

@examples[#:eval ev #:label #f
(define species (~> (load-dataset-table 'iris) (dataset-table-column "species") vector->list))
(remove-duplicates species)
]

@section[#:tag "gs-first-fit"]{A first k-means}

@racket[kmeans] makes an @tech{estimator}, and @racket[fit] fits it on the
flowers. The fit holds three centroids, and @tech{labels} giving each flower
its nearest one:

@examples[#:eval ev #:label #f
(define fitted (fit (kmeans #:n-clusters 3) flowers))
fitted
(define labels (~> fitted kmeans-model kmeans-result-labels device-vector->list))
(length labels)
]

@section[#:tag "gs-species"]{Clusters and species}

k-means never sees the species, and its cluster numbers are arbitrary.
@racket[crosstab], one of the helpers of @secref["examples"], counts each
cluster's flowers by species, one row per cluster, in an order that does not
depend on the cluster numbers:

@examples[#:eval ev #:label #f
(crosstab labels species)
]

Setosa is a cluster of its own. Versicolor and virginica overlap in these
measurements: 14 virginica flowers share the versicolor cluster, and 2
versicolor the virginica one, so 134 of the 150 flowers are in a cluster
whose majority is their species.

@section[#:tag "gs-predict"]{New flowers}

@racket[predict] assigns new rows to the nearest centroid. Naming each
cluster by its majority species turns the cluster numbers into species:

@examples[#:eval ev #:label #f
(define cluster-species
  (for/hash ([label (in-list (remove-duplicates labels))])
    (define names
      (for/list ([sample-label (in-list labels)]
                 [name (in-list species)]
                 #:when (= sample-label label))
        name))
    (values label (~> names (group-by values _) (argmax length _) first))))
(define new-flowers
  (list*->device-matrix '((5.0 3.4 1.5 0.2) (5.9 2.8 4.3 1.3) (6.6 3.0 5.6 2.0))
                        #:dtype 'float32))
(define new-labels (device-vector->list (predict fitted new-flowers)))
(map (curry hash-ref cluster-species) new-labels)
]

Each new row is near the mean of one species, and lands in that species'
cluster. @secref["ex-kmeans"] shows where k-means does less well.

@section[#:tag "gs-not-yet"]{What is not there yet}

Some conveniences have no binding yet, and each has an open issue: a
conversion from a @tt{datasets} table to a device matrix, feature scaling,
clustering metrics, and a spectral clustering example on clusters that are
not convex. Until then the guide's helpers, in @secref["examples"], stand in
for the first three.

@(close-eval ev)
