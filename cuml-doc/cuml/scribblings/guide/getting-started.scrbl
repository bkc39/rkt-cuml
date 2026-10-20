#lang scribble/manual
@(require "../utils.rkt")

@(define ev (make-cuml-eval))

@title[#:tag "getting-started" #:style 'toc]{Getting started}

This chapter copies Fisher's iris flowers to the GPU, clusters them with
k-means, reads the clusters against the species and assigns new flowers to
clusters.

@local-table-of-contents[]

@section[#:tag "gs-installing"]{Installing}

@commandline{raco pkg install cuml datasets}

@tt{datasets} supplies this guide's data, and cuML needs an NVIDIA GPU. Load
both, with @racketmodname[raft] for device arrays and @racketmodname[raft/compat]
for copying a matrix to the device:

@examples[#:eval ev #:label #f
(require cuml datasets raft raft/compat)
]

@section[#:tag "gs-data"]{Data on the device}

The iris data measures 150 flowers of three species, 50 each, in four
columns. @racket[load-dataset] with @racket[#:format 'matrix] gives the
measurements as a @tt{math/matrix} matrix, and
@racket[matrix->device-matrix] copies it to the GPU as a @tech{sample matrix},
one flower per row:

@examples[#:eval ev #:label #f
(define measurements '("sepal-length" "sepal-width" "petal-length" "petal-width"))
(define X
  (~> (load-dataset 'iris #:format 'matrix #:columns measurements)
      (matrix->device-matrix #:dtype 'float32)))
(shape X)
]

The matrix lives on the calling thread's @racket[(current-device-resources)],
as do the results of every call below. The species stay on the host:

@examples[#:eval ev #:label #f
(define species
  (~> (load-dataset 'iris #:format 'table) (dataset-table-column "species") vector->list))
(remove-duplicates species)
]

@section[#:tag "gs-first-fit"]{A first k-means}

@racket[kmeans] makes an @tech{estimator}, and @racket[fit] fits it on
@racket[X]. The fit holds three centroids, and @tech{labels} giving each
flower its nearest one:

@examples[#:eval ev #:label #f
(define fitted (fit (kmeans #:n-clusters 3) X))
fitted
(define labels (~> fitted kmeans-model kmeans-result-labels device-vector->list))
(length labels)
]

@section[#:tag "gs-species"]{Clusters and species}

k-means never sees the species, and its cluster numbers are arbitrary. To
compare, count each cluster's flowers by species. Every evaluator in this
guide has four helpers, written in plain Racket: @racket[crosstab] gives one
row per cluster and one column per class, in the order the classes first
appear, the rows sorted by their largest column so that the table does not
depend on the cluster numbers; @racket[agreement] sums each row's largest
count; @racket[sizes] lists the cluster sizes, largest first; and
@racket[standardized-rows] rescales each column of a @tt{datasets} table to
mean 0 and standard deviation 1, as rows.

@page-helpers-block

@examples[#:eval ev #:label #f
(crosstab labels species)
(agreement (crosstab labels species))
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
  (for/hash ([cluster (in-list (group-by car (map cons labels species)))])
    (values (caar cluster) (~> (map cdr cluster) (group-by values _) (argmax length _) car))))
(define new-flowers
  (list*->device-matrix '((5.0 3.4 1.5 0.2) (5.9 2.8 4.3 1.3) (6.6 3.0 5.6 2.0))
                        #:dtype 'float32))
(define new-labels (device-vector->list (predict fitted new-flowers)))
(for/list ([label (in-list new-labels)])
  (hash-ref cluster-species label))
]

Each new row is near the mean of one species, and lands in that species'
cluster. @secref["ex-kmeans"] shows where k-means does less well.

@(close-eval ev)
