#lang scribble/manual
@(require "../utils.rkt")

@title[#:tag "examples" #:style 'toc]{Examples}

One chapter per algorithm, each on a real dataset from @tt{datasets} with
known classes or known structure, and each saying why the algorithm suits that
data or does not. The clusters are read with the helpers of
@secref["gs-species"], so no number in the prose depends on how a run numbers
its clusters.

@local-table-of-contents[]

@include-section["examples/kmeans.scrbl"]
@include-section["examples/dbscan.scrbl"]
@include-section["examples/single-linkage.scrbl"]
@include-section["examples/spectral.scrbl"]
