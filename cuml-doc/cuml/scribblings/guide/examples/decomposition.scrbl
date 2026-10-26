#lang scribble/manual
@(require "../../utils.rkt")

@(define ev (make-cuml-eval))

@title[#:tag "ex-decomposition"]{Decomposition: wine in two dimensions}

Principal component analysis (PCA) finds orthogonal directions, its
components, in decreasing order of the data's variance along them. The first
few keep most of the variance in fewer columns, for a plot or a clustering.
Truncated SVD does the same without centring the data.

@section[#:tag "ex-decomposition-data"]{The data}

The wine measurements of @secref["ex-kmeans"]. @racket[renumbered] numbers
the cultivars 0 to 2 for @racket[adjusted-rand-score]:

@examples[#:eval ev #:label #f
(define wine (load-dataset-table 'wine))
(define measurements (features 'wine "cultivar"))
(define cultivar (dataset-column 'wine "cultivar"))
(define cultivar-ids (list->device-vector (renumbered cultivar) #:dtype 'int32))
(define raw (table->device-matrix wine #:columns measurements))
(define-values (wine-scaler scaled) (standard-scaler-fit-transform raw))
(define (rounded x) (~r x #:precision 3))
]

@section[#:tag "ex-decomposition-scaling"]{Scale first}

PCA ranks directions by variance in the data's own units. @racket[leading]
names each component's largest entries:

@examples[#:eval ev #:label #f
(define (leading result n)
  (for/list ([component (in-list (device-matrix->list* (pca-result-components result)))])
    (~> (map cons component measurements)
        (sort > #:key (compose abs car))
        (take n)
        (map cdr _))))
(define raw-fit (pca-fit raw #:n-components 2))
(~> raw-fit pca-result-explained-variance-ratio device-vector->list (map rounded _))
(leading raw-fit 1)
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
(define cumulative (for/list ([k (in-range 1 14)]) (apply + (take ratios k))))
(rounded (second cumulative))
(map (lambda (share) (add1 (index-where cumulative (curry <= share)))) '(0.8 0.9))
(line-plot (range 1 14) cumulative #:x-label "components" #:y-label "explained variance")
(leading scaled-fit 2)
]

Two components keep 0.554 of the variance, five 80% and eight 90%. The first
is led by flavanoids and total phenols, the second by colour intensity and
alcohol.

@section[#:tag "ex-decomposition-kmeans"]{k-means on two components}

@racket[fit-transform] projects the wines on the first two components;
@racket[#:whiten] also rescales each to variance 1. k-means on each, scored
against the cultivars:

@examples[#:eval ev #:label #f
(define projected (fit-transform (pca #:n-components 2) scaled))
(define whitened (fit-transform (pca #:n-components 2 #:whiten #t) scaled))
(for/list ([samples (list scaled projected whitened)])
  (~> (fit-predict (kmeans #:n-clusters 3) samples) (adjusted-rand-score cultivar-ids _) rounded))
]

Two components cluster better than all 13 columns, 0.912 against 0.864;
whitened, they find the same clusters.

The wines in the plane of the two components, each clustering in the colour of
the cultivar it shares the most wines with:

@examples[#:eval ev #:label #f
(define plane (device-matrix->list* projected))
(class-figure (map first plane) (map second plane)
              (list (list "13 columns" (clusters-of (kmeans #:n-clusters 3) scaled cultivar))
                    (list "2 components" (clusters-of (kmeans #:n-clusters 3) projected cultivar)))
              cultivar
              #:class-title "cultivar"
              #:names (map (curry format "cultivar ~a") (remove-duplicates cultivar))
              #:x-label "first component"
              #:y-label "second component")
]

The first component runs from cultivar 3 to cultivar 1, and cultivar 2 lies
below both on the second. The 13 columns misplace 8 wines, all of cultivar 2;
the two components misplace 5, four of them cultivar 2.

@section[#:tag "ex-decomposition-tsvd"]{Truncated SVD}

Truncated SVD factors the data as it is, without subtracting the column means.
The standardized columns have mean 0, so there it finds PCA's singular values
and ratios:

@examples[#:eval ev #:label #f
(define svd-fit (tsvd-fit scaled #:n-components 2))
(~> (list (tsvd-result-singular-values svd-fit) (tsvd-result-explained-variance-ratio svd-fit))
    (map (compose (curry map rounded) device-vector->list) _))
(~> (list (pca-result-singular-values scaled-fit) (pca-result-explained-variance-ratio scaled-fit))
    (map (compose (curry map rounded) (curryr take 2) device-vector->list) _))
]

Use PCA to reduce correlated measurements to a few columns before a plot or a
clustering. Truncated SVD gives the same on centred data, and leaves data
uncentred where the means must stay.

@(close-eval ev)
