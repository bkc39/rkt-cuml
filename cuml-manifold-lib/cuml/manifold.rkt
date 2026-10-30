#lang racket/base

(require "manifold/private/tsne.rkt"
         "manifold/private/umap.rkt")

(provide (except-out (all-from-out "manifold/private/tsne.rkt") tsne-fit-transform/who tsne-fit/who)
         (except-out (all-from-out "manifold/private/umap.rkt")
                     umap-fit-transform/who
                     umap-fit/who
                     umap-transform/who))
