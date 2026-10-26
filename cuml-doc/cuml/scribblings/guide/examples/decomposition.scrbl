#lang scribble/manual
@(require "../../utils.rkt")

@(define ev (make-cuml-eval))

@title[#:tag "ex-decomposition"]{Decomposition: wine in two dimensions}

@section[#:tag "ex-decomposition-data"]{The data}

The wine measurements of @secref["ex-kmeans"]:

@examples[#:eval ev #:label #f
(define wine (load-dataset-table 'wine))
(define measurements (features 'wine "cultivar"))
(define cultivar (dataset-column 'wine "cultivar"))
(define cultivar-ids (class-ids cultivar))
(define raw (table->device-matrix wine #:columns measurements))
(define-values (wine-scaler scaled) (standard-scaler-fit-transform raw))
]

@section[#:tag "ex-decomposition-scaling"]{Scale first}

PCA ranks directions by variance in the data's own units:

@examples[#:eval ev #:label #f
(define raw-fit (pca-fit raw #:n-components 2))
(~> raw-fit pca-result-explained-variance-ratio device-vector->list (map rounded _))
(leading raw-fit measurements 1)
]

On raw measurements the first component is proline, which alone explains
0.998 of the variance, and the second magnesium: the two widest columns.
Standardize first unless the units are comparable.

@section[#:tag "ex-decomposition-variance"]{Explained variance}

Standardized, each measurement has variance 1, and the variance spreads over
the components:

@examples[#:eval ev #:label #f
(define scaled-fit (pca-fit scaled))
(define ratios (~> scaled-fit pca-result-explained-variance-ratio device-vector->list))
(map rounded (take ratios 3))
(~> ratios cumulative second rounded)
(map (curry components-for ratios) '(0.8 0.9))
(line-plot (inclusive-range 1 (length ratios))
           (cumulative ratios)
           #:x-label "components"
           #:y-label "explained variance")
(leading scaled-fit measurements 2)
]

Two components keep 0.554 of the variance, five 80% and eight 90%. The first
is led by flavanoids and total phenols, the second by colour intensity and
alcohol.

@section[#:tag "ex-decomposition-kmeans"]{k-means on two components}

@racket[fit-transform] projects the wines on the first two components;
@racket[#:whiten] also rescales each to variance 1. k-means, best of ten
starts, on each, scored against the cultivars:

@examples[#:eval ev #:label #f
(define projected (fit-transform (pca #:n-components 2) scaled))
(define whitened (fit-transform (pca #:n-components 2 #:whiten #t) scaled))
(define clusterer (kmeans #:n-clusters 3 #:n-init 10))
(~> (list scaled projected whitened)
    (map (curry fit-predict clusterer) _)
    (map (curry adjusted-rand-score cultivar-ids) _)
    (map rounded _))
]

Two components cluster about as well as all 13 columns, 0.895 against 0.897;
whitened, better: 0.912.

The wines in the plane of the two components, each clustering in the colour of
the cultivar it shares the most wines with:

@examples[#:eval ev #:label #f
(class-figure projected
              (list (list "13 columns" (clusters-of clusterer scaled cultivar))
                    (list "2 components" (clusters-of clusterer projected cultivar)))
              cultivar
              #:class-title "cultivar"
              #:x-label "first component"
              #:y-label "second component")
]

The first component runs from cultivar 3 to cultivar 1, and cultivar 2 lies
below both on the second. Each clustering misplaces 6 wines, all of
cultivar 2.

@section[#:tag "ex-decomposition-tsvd"]{Truncated SVD}

Truncated SVD factors the data as it is, without subtracting the column means.
The standardized columns have mean 0, so there it finds PCA's singular values
and ratios:

@examples[#:eval ev #:label #f
(define svd-fit (tsvd-fit scaled #:n-components 2))
(spectrum svd-fit 2)
(spectrum scaled-fit 2)
]

@(close-eval ev)
