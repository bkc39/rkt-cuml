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
and leaves a constant column at 0. @racket[dataset-matrix] and
@racket[standardized-matrix] copy a dataset's columns to the GPU, one sample
per row. @racket[labels-of] fits an estimator and copies its labels to the
host. No number in the prose depends on how a run numbers its clusters.

@local-table-of-contents[]

@include-section["examples/kmeans.scrbl"]
@include-section["examples/dbscan.scrbl"]
@include-section["examples/single-linkage.scrbl"]
@include-section["examples/spectral.scrbl"]
