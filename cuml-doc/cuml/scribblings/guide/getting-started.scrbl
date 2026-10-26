#lang scribble/manual
@(require "../utils.rkt")

@(define ev (make-cuml-eval))

@title[#:tag "getting-started" #:style 'toc]{Getting started}

This chapter copies Fisher's iris flowers to the GPU, clusters them with
k-means, reads the clusters against the species and assigns new flowers to
clusters.

@local-table-of-contents[]

@section[#:tag "gs-installing"]{Installing}

@commandline{raco pkg install --auto cuml datasets-core plot-lib threading}

The packages require Racket 9.3 or later, and cuML an NVIDIA GPU.
@tt{datasets-core} supplies this guide's data, @tt{plot-lib} its figures and
@tt{threading} its @racket[~>]. Every example in the guide starts from this
require and the helpers in @secref["examples"]:

@page-require-block

@section[#:tag "gs-data"]{Data on the device}

The iris data (R. A. Fisher, 1936, from the UCI Machine Learning Repository,
@hyperlink["https://doi.org/10.24432/C56C76"]{doi:10.24432/C56C76}, under
CC BY 4.0) measures 150 flowers of three species, 50 each, in four columns.
@racket[load-dataset-table] loads the table, and @racket[table->device-matrix]
copies its measurement columns to the GPU as a @racket['float32]
@tech{sample matrix}, one flower per row:

@examples[#:eval ev #:label #f
(define measurements '("sepal-length" "sepal-width" "petal-length" "petal-width"))
(define flowers (table->device-matrix (load-dataset-table 'iris) #:columns measurements))
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
@racket[counts-of], a helper of @secref["examples"] over
@racket[contingency-matrix], counts each species' flowers, one row per
species, in each cluster, one column per cluster:

@examples[#:eval ev #:label #f
(define species-ids (class-ids species))
(define cluster-labels (~> fitted kmeans-model kmeans-result-labels))
(counts-of species-ids cluster-labels)
(rounded (adjusted-rand-score species-ids cluster-labels))
]

Setosa is a cluster of its own. Versicolor and virginica overlap in these
measurements: 14 virginica flowers share the versicolor cluster, and 2
versicolor the virginica one. The adjusted Rand index is 0.73.

@racket[class-figure] draws the flowers by petal length and width, by cluster
and by species. @racket[matched-labels] gives each cluster the colour and
marker of the species it shares the most flowers with, so a flower drawn
differently in the two panels is one of the 16 in another species' cluster.
The crosses are the centroids.

@examples[#:eval ev #:label #f
(define petal-length (dataset-column 'iris "petal-length"))
(define petal-width (dataset-column 'iris "petal-width"))
(define centroids
  (~> fitted kmeans-model kmeans-result-centroids device-matrix->list* (map (curryr drop 2) _)))
(class-figure petal-length petal-width
              (list (list "k-means" (matched-labels labels species) centroids))
              species
              #:class-title "species"
              #:x-label "petal length (cm)"
              #:y-label "petal width (cm)")
]

Setosa's petals are shorter than any other flower's. The 16 have petals 4.5 to
5.1 cm long, the lengths both versicolor and virginica reach.

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

Some conveniences have no binding yet, and each has an open issue: encoding
class names as label numbers, which @racket[class-ids] stands in for, and a
spectral clustering example on clusters that are not convex.

@(close-eval ev)
