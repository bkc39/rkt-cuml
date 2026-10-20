#lang scribble/manual
@(require "utils.rkt")

@title[#:tag "guide" #:style 'toc]{User guide}

@secref["getting-started"] clusters Fisher's iris flowers on the GPU and
reads the clusters against the species. @secref["concepts"] covers what every
procedure shares, and the @secref["examples"] take each algorithm to a real
dataset. The @secref["reference"] documents each binding.

@local-table-of-contents[]

@include-section["guide/getting-started.scrbl"]
@include-section["guide/concepts.scrbl"]
@include-section["guide/examples.scrbl"]
