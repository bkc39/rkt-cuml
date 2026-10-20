#lang scribble/manual
@(require "../../utils.rkt")

@(define ev (make-cuml-eval))

@title[#:tag "ex-spectral"]{Spectral clustering: breast tumours}

Spectral clustering joins each sample to its @racket[n-neighbors] nearest,
embeds the samples by the eigenvectors of that graph's Laplacian, and runs
k-means on the embedding. It cuts the graph where it is sparse, so a cluster
need not be round, only connected. It suits clusters of uneven shape and
spread, and costs an eigensolve.

@section[#:tag "ex-spectral-data"]{The data}

The breast-cancer data describes 569 tumours, malignant or benign, by 30
measurements of their cell nuclei. Leave out the diagnosis and the
identifier, and standardize the rest:

@examples[#:eval ev #:label #f
(define tumours (load-dataset 'breast-cancer #:format 'table))
(define features (remove* '("diagnosis" "id") (dataset-table-names tumours)))
(define diagnosis (vector->list (dataset-table-column tumours "diagnosis")))
(sizes diagnosis)
(define X
  (~> (load-dataset 'breast-cancer #:format 'table #:columns features)
      standardized-rows
      (list*->device-matrix #:dtype 'float32)))
]

@section[#:tag "ex-spectral-fit"]{Against k-means}

@examples[#:eval ev #:label #f
(define (table-of estimator)
  (~> (fit-predict estimator X) device-vector->list (crosstab diagnosis)))
(define kmeans-table (table-of (kmeans #:n-clusters 2)))
(define spectral-table (table-of (spectral-clustering #:n-clusters 2)))
kmeans-table
spectral-table
(map agreement (list kmeans-table spectral-table))
]

The benign tumours are many and alike, the malignant ones fewer and more
varied. k-means' boundary lies halfway between two centres and cuts into the
spread class: it puts 36 malignant tumours in the benign cluster and 18 benign
ones in the malignant cluster. Following the neighbour graph, spectral
clustering misplaces 30 and 6.

@section[#:tag "ex-spectral-neighbours"]{The neighbour graph}

@racket[n-neighbors] sets the graph. Three neighbours are too few; from five
on, the agreement changes little:

@examples[#:eval ev #:label #f
(for/list ([k (in-list '(3 5 10 30))])
  (list k (agreement (table-of (spectral-clustering #:n-clusters 2 #:n-neighbors k)))))
]

Spectral clustering differs most from k-means on clusters that are not
convex, such as rings or crescents; on these compact classes its gain is
modest.

@(close-eval ev)
