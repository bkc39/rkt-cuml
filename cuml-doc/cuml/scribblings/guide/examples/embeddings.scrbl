#lang scribble/manual
@(require "../../utils.rkt")

@(define ev (make-cuml-eval))

@title[#:tag "ex-embeddings"]{Embeddings: handwritten digits in two dimensions}

UMAP and t-SNE place samples in two dimensions so that neighbours stay
neighbours. Clusters that run together in many dimensions can come apart
there, dense enough for HDBSCAN to find.

@section[#:tag "ex-embeddings-data"]{The data}

The digits data (E. Alpaydin and C. Kaynak, 1998, from the UCI Machine Learning
Repository, @hyperlink["https://doi.org/10.24432/C50P49"]{doi:10.24432/C50P49},
under CC BY 4.0) holds 1797 handwritten digits, each an 8×8 grid of pixel
counts from 0 to 16:

@examples[#:eval ev #:label #f
(define digits (load-dataset-table 'digits))
(define pixels (features 'digits "digit"))
(define digit (dataset-column 'digits "digit"))
(define digit-ids (class-ids digit))
(define images (table->device-matrix digits #:columns pixels))
(list (length pixels) (sizes digit))
]

@section[#:tag "ex-embeddings-raw"]{HDBSCAN on the pixels}

A cluster needs 60 digits, a third of a digit's samples, and a core sample
10 neighbours. @racket[scored] gives the adjusted Rand index against the
digits, the cluster sizes and the noise:

@examples[#:eval ev #:label #f
(define clusterer (hdbscan #:min-cluster-size 60 #:min-samples 10))
(scored clusterer digit-ids images)
]

In 64 dimensions the digits' densities run together: six clusters, one of
461 digits mixing 1s, 3s, 5s, 8s and 9s, and 633 digits noise.

@section[#:tag "ex-embeddings-umap"]{UMAP, then HDBSCAN}

UMAP at its defaults, 15 neighbours and seed 0, embeds the digits in two
dimensions; HDBSCAN clusters the embedding with the same settings:

@examples[#:eval ev #:label #f
(define embedded (fit-transform (umap) images))
(scored clusterer digit-ids embedded)
]

Ten clusters, scoring 0.899 against 0.284 on the pixels, and 6 digits noise.
Each cluster in the colour of the digit it holds most of:

@examples[#:eval ev #:label #f
(class-figure embedded
              (list (list "HDBSCAN" (clusters-of clusterer embedded digit)))
              digit
              #:class-title "digit")
]

Each digit has a cluster of its own. Of the 91 digits the clusters misplace,
66 are 1s or 9s.

@section[#:tag "ex-embeddings-tsne"]{t-SNE}

t-SNE, at its defaults and seed 0, draws the digits apart too:

@examples[#:eval ev #:label #f
(define pictured (fit-transform (tsne) images))
(scored clusterer digit-ids pictured)
(~> (list (fit-transform (pca #:n-components 2) images) embedded pictured)
    (map (curry trustworthiness images) _)
    (map rounded _))
(class-figure pictured
              (list (list "HDBSCAN" (clusters-of clusterer pictured digit)))
              digit
              #:class-title "digit")
]

@racket[trustworthiness] is 1 when each digit's 5 nearest neighbours in the
picture are among its nearest in the pixels. t-SNE keeps them as well as
UMAP, 0.992 against 0.988, and both far better than the first two principal
components, 0.83. But t-SNE does not keep density: HDBSCAN on it scores
0.745 and calls 187 digits noise. Cluster on UMAP's embedding; t-SNE draws
the neighbourhoods.

@(close-eval ev)
