#lang scribble/manual
@(require "../../utils.rkt")

@(define ev (make-cuml-eval))

@title[#:tag "ex-kmeans"]{k-means: wine cultivars}

k-means places @racket[n-clusters] centroids and gives each sample the
nearest, by Euclidean distance. It suits compact, roughly round clusters of
similar spread whose number is known. Every feature counts in its own units.

@section[#:tag "ex-kmeans-data"]{The data}

The wine data (S. Aeberhard and M. Forina, 1991, from the UCI Machine Learning
Repository, @hyperlink["https://doi.org/10.24432/C5PC7J"]{doi:10.24432/C5PC7J},
under CC BY 4.0) holds 13 chemical measurements of 178 wines from three
cultivars:

@examples[#:eval ev #:label #f
(define wine (load-dataset-table 'wine))
(define measurements (features 'wine "cultivar"))
(define cultivar (dataset-column 'wine "cultivar"))
(sizes cultivar)
]

The measurements have very different ranges:

@examples[#:eval ev #:label #f
(for/list ([name (in-list '("proline" "magnesium" "hue"))])
  (define column (dataset-column 'wine name))
  (list name (apply min column) (apply max column)))
]

@section[#:tag "ex-kmeans-raw"]{Raw measurements}

@racket[table-of] clusters wines into three and counts each cluster's
cultivars:

@examples[#:eval ev #:label #f
(define (table-of wines)
  (crosstab (labels-of (kmeans #:n-clusters 3) wines) cultivar))
(define raw-table (table-of (table->device-matrix wine #:columns measurements)))
raw-table
(agreement raw-table)
]

125 of the 178 wines are in a cluster whose majority is their cultivar.
Proline, in the hundreds, decides the distances: on proline alone, k-means
finds the same clusters.

@examples[#:eval ev #:label #f
(equal? raw-table (table-of (table->device-matrix wine #:columns '("proline"))))
]

@section[#:tag "ex-kmeans-standardized"]{Standardized measurements}

With each column rescaled to standard deviation 1, every measurement counts
alike:

@examples[#:eval ev #:label #f
(define scaled (standardized-matrix 'wine measurements))
(define scaled-table (table-of scaled))
scaled-table
(agreement scaled-table)
]

170 of 178 now agree. Scale the features before k-means unless their units
are comparable.

Proline against flavanoids, in their own units, with each cluster in the colour
of the cultivar it shares the most wines with:

@examples[#:eval ev #:label #f
(define proline (dataset-column 'wine "proline"))
(define flavanoids (dataset-column 'wine "flavanoids"))
(define raw (table->device-matrix wine #:columns measurements))
(class-figure proline flavanoids
              (list (list "raw" (clusters-of (kmeans #:n-clusters 3) raw cultivar))
                    (list "standardized" (clusters-of (kmeans #:n-clusters 3) scaled cultivar)))
              cultivar
              #:class-title "cultivar"
              #:names (map (curry format "cultivar ~a") (remove-duplicates cultivar))
              #:x-label "proline"
              #:y-label "flavanoids")
]

On raw measurements the clusters are bands of proline, whatever the
flavanoids; standardized, they follow the cultivars.

@section[#:tag "ex-kmeans-k"]{Choosing the number of clusters}

k-means needs @racket[n-clusters]. @racket[score] is the negated inertia, the
sum of squared distances to the nearest centroid, which grows towards 0 with
every added cluster; where its gains shrink is a hint. Ten seedings per fit
keep a poor local optimum from hiding the trend:

@examples[#:eval ev #:label #f
(define scores
  (for/list ([k (in-range 1 7)])
    (define fitted (fit (kmeans #:n-clusters k #:n-init 10) scaled))
    (list k (exact-round (score fitted scaled)))))
scores
(line-plot (map first scores) (map second scores) #:x-label "clusters" #:y-label "score")
]

The score gains 655 and 381 from the second and third clusters, and only 102
from the fourth: three is where the line bends, as the cultivars say.

@(close-eval ev)
