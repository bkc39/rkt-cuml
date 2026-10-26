#lang scribble/manual
@(require "../utils.rkt")

@title[#:tag "examples" #:style 'toc]{Examples}

One chapter per algorithm or family, each on a real dataset from
@tt{datasets-core} with known classes or known structure, and each saying why
the algorithm suits that data or does not. Each chapter names its data's source and licence, and
the @hyperlink["https://docs.racket-lang.org/datasets-core/datasets-catalog.html"]{datasets-core
catalog} gives its provenance.

Every chapter of this guide runs these helpers, written in plain Racket:

@page-helpers-block

@racket[sizes] lists the cluster sizes, largest first, leaving out noise,
label @racket[-1]. @racket[labels-of] fits an estimator and copies its labels
to the host. No number in the prose depends on how a run numbers its
clusters.

@racket[cluster-plot] draws samples by two columns, one colour and marker per
label id 0 to 7, noise gray; with @racket[#:core], core samples are filled and
the others hollow, and @racket[#:centroids] adds crosses. @racket[class-figure]
draws clusterings beside the classes, @racket[sweep-figure] one clustering per
setting, each with a legend; @racket[line-plot] draws one line.
@racket[renumbered] numbers values in the order they first appear, keeping
@racket[-1]. @racket[matched-labels] gives each cluster the number of the class
it shares the most samples with, largest overlap first, and numbers the rest
after the classes; the classes can be another clustering's labels.
@racket[clusters-of] fits an estimator and matches its labels to classes.
@racket[class-ids] copies class names to the device as the numbers
@racket[renumbered] gives them, for the metrics; @racket[counts-of] counts
each class's samples in each cluster, and @racket[rounded] prints a score to
three places. @racket[leading] names each principal component's largest
entries, @racket[cumulative] sums explained-variance ratios,
@racket[components-for] counts the components that reach a share, and
@racket[columns-of] copies a matrix's columns to the host.

@local-table-of-contents[]

@include-section["examples/kmeans.scrbl"]
@include-section["examples/dbscan.scrbl"]
@include-section["examples/single-linkage.scrbl"]
@include-section["examples/spectral.scrbl"]
@include-section["examples/hdbscan.scrbl"]
@include-section["examples/decomposition.scrbl"]
