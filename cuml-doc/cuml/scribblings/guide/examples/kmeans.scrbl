#lang scribble/manual
@(require "../../utils.rkt")

@(define ev (make-cuml-eval))

@title[#:tag "ex-kmeans"]{k-means: wine cultivars}

k-means places @racket[n-clusters] centroids and gives each sample the
nearest, by Euclidean distance. It suits compact, roughly round clusters of
similar spread whose number is known. Every feature counts in its own units.

@section[#:tag "ex-kmeans-data"]{The data}

The wine data holds 13 chemical measurements of 178 wines from three
cultivars:

@examples[#:eval ev #:label #f
(define wine (load-dataset 'wine #:format 'table))
(define features (remove "cultivar" (dataset-table-names wine)))
(define cultivar (vector->list (dataset-table-column wine "cultivar")))
(sizes cultivar)
]

The measurements have very different ranges:

@examples[#:eval ev #:label #f
(for/list ([name (in-list '("proline" "magnesium" "hue"))])
  (define column (vector->list (dataset-table-column wine name)))
  (list name (apply min column) (apply max column)))
]

@section[#:tag "ex-kmeans-raw"]{Raw measurements}

@examples[#:eval ev #:label #f
(define raw
  (~> (load-dataset 'wine #:format 'matrix #:columns features)
      (matrix->device-matrix #:dtype 'float32)))
(define raw-table
  (~> (fit-predict (kmeans #:n-clusters 3) raw) device-vector->list (crosstab cultivar)))
raw-table
(agreement raw-table)
]

125 of the 178 wines are in a cluster whose majority is their cultivar.
Proline, in the hundreds, decides the distances: on proline alone, k-means
finds the same clusters.

@examples[#:eval ev #:label #f
(define proline
  (~> (load-dataset 'wine #:format 'matrix #:columns '("proline"))
      (matrix->device-matrix #:dtype 'float32)))
(equal? raw-table
        (~> (fit-predict (kmeans #:n-clusters 3) proline) device-vector->list (crosstab cultivar)))
]

@section[#:tag "ex-kmeans-standardized"]{Standardized measurements}

With each column rescaled to standard deviation 1, every measurement counts
alike:

@examples[#:eval ev #:label #f
(define scaled
  (~> (load-dataset 'wine #:format 'table #:columns features)
      standardized-rows
      (list*->device-matrix #:dtype 'float32)))
(define scaled-table
  (~> (fit-predict (kmeans #:n-clusters 3) scaled) device-vector->list (crosstab cultivar)))
scaled-table
(agreement scaled-table)
]

170 of 178 now agree. Scale the features before k-means unless their units
are comparable.

@section[#:tag "ex-kmeans-k"]{Choosing the number of clusters}

k-means needs @racket[n-clusters]. @racket[score] is the negated inertia, the
sum of squared distances to the nearest centroid, which grows towards 0 with
every added cluster; where its gains shrink is a hint:

@examples[#:eval ev #:label #f
(for/list ([k (in-range 1 7)])
  (define fitted (fit (kmeans #:n-clusters k) scaled))
  (list k (exact-round (score fitted scaled))))
]

The score gains 654 and 380 from the second and third clusters, and only 68
from the fourth: three is where the gains shrink, as the cultivars say.

@(close-eval ev)
