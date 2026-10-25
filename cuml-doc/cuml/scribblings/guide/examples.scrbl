#lang scribble/manual
@(require "../utils.rkt")

@title[#:tag "examples" #:style 'toc]{Examples}

One chapter per algorithm, each on a real dataset from @tt{datasets-core}
with known classes or known structure, and each saying why the algorithm suits
that data or does not. Each chapter names its data's source and licence, and
the @hyperlink["https://docs.racket-lang.org/datasets-core/datasets-catalog.html"]{datasets-core
catalog} gives its provenance.

Every chapter of this guide runs these helpers, written in plain Racket:

@page-helpers-block

@racket[crosstab] gives one row per cluster and one column per class, in the
order the classes first appear, with the rows sorted by their largest column;
@racket[agreement] sums each row's largest count; @racket[sizes] lists the
cluster sizes, largest first. All three leave out noise, label @racket[-1].
@racket[standardized] rescales each column to mean 0 and standard deviation 1,
and leaves a constant column at 0. @racket[standardized-matrix] copies a
dataset's columns to the GPU standardized, one sample per row. @racket[labels-of] fits an estimator and copies its labels to the
host. No number in the prose depends on how a run numbers its clusters.

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

@local-table-of-contents[]

@include-section["examples/kmeans.scrbl"]
@include-section["examples/dbscan.scrbl"]
@include-section["examples/single-linkage.scrbl"]
@include-section["examples/spectral.scrbl"]
@include-section["examples/hdbscan.scrbl"]
