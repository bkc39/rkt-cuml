#lang scribble/manual
@(require scribble/example
          (for-label cuml
                     racket/base
                     racket/contract/base
                     racket/match
                     raft
                     threading))

@(define ev (make-base-eval #:lang 'racket/base '(require cuml raft racket/match threading)))

@title[#:tag "spectral"]{Spectral clustering}

@declare-exporting[cuml/cluster cuml]

@examples[#:eval ev
(define G
  (list*->device-matrix (for*/list ([centre (in-list '(0 10 20))]
                                    [i (in-range 8)])
                          (list (+ centre (* 0.5 (quotient i 4))) (* 0.5 (remainder i 4))))
                        #:dtype 'float32))
(~> (spectral-clustering-fit G #:n-clusters 3 #:n-neighbors 5)
    spectral-clustering-result-labels
    device-vector->list)
]

@defstruct*[spectral-clustering-result ([labels device-vector?])]{

The result of @racket[spectral-clustering-fit] and
@racket[spectral-clustering-fit-graph]: each sample's cluster, @racket['int32],
from 0 to k−1, numbered by k-means, so a label can go unused. Runs can
renumber the labels: compare partitions, not label values.}

@defproc[(spectral-clustering-fit [X device-matrix?]
                                  [#:n-clusters n-clusters exact-positive-integer? 8]
                                  [#:n-components n-components (or/c #f exact-positive-integer?) #f]
                                  [#:random-state random-state (integer-in 0 18446744073709551615) 0]
                                  [#:n-neighbors n-neighbors (and/c exact-integer? (>=/c 2)) 10]
                                  [#:n-init n-init (integer-in 1 2147483647) 10]
                                  [#:eigen-tol eigen-tol (or/c 'auto (and/c real? (>=/c 0))) 'auto]
                                  [#:resources resources device-resources? (current-device-resources)])
         spectral-clustering-result?]{

Clusters the rows of @racket[X], a row-major @racket['float32] matrix with at
least 4 rows, from the normalized Laplacian of the graph that joins each
sample to its min(@racket[n-neighbors], n−1) nearest samples, itself
included. The first @racket[n-components] eigenvectors, by default
min(@racket[n-clusters], ⌊(n−1)/3⌋), are clustered by k-means with
@racket[n-init] runs, seeded by @racket[random-state]. An explicit
@racket[n-components] must lie in [1, ⌊(n−1)/3⌋], and more components than
clusters can blur the partition. Below 7 samples the automatic embedding has
one component, and the labels need not follow the graph: they can split a
connected graph into pieces that are not themselves connected, and return
fewer clusters than asked.

The eigensolver stops at a residual of at most @racket[eigen-tol]
(@racket['auto] is 0) or after 10n iterations, without reporting that it did
not converge, so labels near cluster boundaries can vary between runs. Graphs
with repeated eigenvalues, such as cliques, heavily repeated rows or many
isolated samples, can fail with ``eigensolver couldn't converge''. The labels
live on @racket[resources].}

@defproc[(spectral-clustering-fit-graph [rows device-vector?]
                                        [cols device-vector?]
                                        [vals device-vector?]
                                        [n-samples (integer-in 4 2147483647)]
                                        [#:n-clusters n-clusters exact-positive-integer? 8]
                                        [#:n-components n-components (or/c #f exact-positive-integer?) #f]
                                        [#:random-state random-state (integer-in 0 18446744073709551615) 0]
                                        [#:n-init n-init (integer-in 1 2147483647) 10]
                                        [#:eigen-tol eigen-tol (or/c 'auto (and/c real? (>=/c 0))) 'auto]
                                        [#:resources resources device-resources? (current-device-resources)])
         spectral-clustering-result?]{

Clusters @racket[n-samples] samples from an affinity graph given as
coordinates: entry @racket[i] joins sample @racket[(rows i)] to sample
@racket[(cols i)] with weight @racket[(vals i)]. @racket[rows] and
@racket[cols] are @racket['int32] and @racket[vals] is @racket['float32], all
of one length. The graph must be symmetric, without self-loops, with finite
weights of at least 0. Duplicate entries add, and a sample with no edges gets
some label. Otherwise as @racket[spectral-clustering-fit].

@racket[G]'s groups, joined within each group to the samples at most two away:

@examples[#:eval ev #:label #f
(define-values (rows cols)
  (for*/lists (rows cols)
              ([i (in-range 24)]
               [j (in-range 24)]
               #:when (and (not (= i j))
                           (= (quotient i 8) (quotient j 8))
                           (<= (abs (- i j)) 2)))
    (values i j)))
(~> (spectral-clustering-fit-graph (list->device-vector rows #:dtype 'int32)
                                   (list->device-vector cols #:dtype 'int32)
                                   (list->device-vector (map (lambda (_) 1.0) rows) #:dtype 'float32)
                                   24
                                   #:n-clusters 3)
    spectral-clustering-result-labels
    device-vector->list)
]

A self-loop is refused:

@examples[#:eval ev #:label #f
(eval:error
 (spectral-clustering-fit-graph (list->device-vector '(0 1 2) #:dtype 'int32)
                                (list->device-vector '(1 1 0) #:dtype 'int32)
                                (list->device-vector '(1.0 1.0 1.0) #:dtype 'float32)
                                4
                                #:n-clusters 2))
]}

@bold{Errors.} @racket[spectral-clustering-fit] and
@racket[spectral-clustering-fit-graph] raise @racket[exn:fail:cuml] of kind
@racket['logic], leaving nothing allocated, when:
@itemlist[
 @item{@racket[X] is not a row-major @racket['float32] matrix with at least 4
       rows and a column, or @racket[rows], @racket[cols] and @racket[vals] are
       not vectors of their dtypes and of one length;}
 @item{@racket[X] has more than 2^31-1 elements, its kNN graph would pass
       2^31-1 entries, or the graph's entries and samples together pass
       2^31-1;}
 @item{@racket[n-samples] is less than 4 or more than 2^31-1;}
 @item{@racket[n-clusters] is less than 1 or more than the number of samples;}
 @item{the eigensolver's basis of n × min(n − c, max(2c + 1, 20)) elements,
       for c components, would pass 2^31-1;}
 @item{the k-means++ buffer of (2 + ⌈ln k⌉) × n elements, for k clusters,
       would pass 2^31-1;}
 @item{@racket[n-components] is outside [1, ⌊(n−1)/3⌋], @racket[n-neighbors]
       is less than 2, @racket[n-init] is less than 1, or @racket[eigen-tol] is
       negative, NaN or past the largest @racket['float32];}
 @item{a row of @racket[X] holds a NaN or an infinity, or its squared norm
       passes 2^125;}
 @item{an entry's index is outside [0, @racket[n-samples]), the entry is a
       self-loop, or its weight is negative, not finite or subnormal: the
       first such entry is named;}
 @item{the weights sum past the largest @racket['float32], or none is
       positive.}]

cuML's own failures, such as an eigensolver that could not converge, raise
@racket[exn:fail:cuml] of kind @racket['generic] naming the procedure, with
nothing allocated, and running out of device memory has kind
@racket['out-of-memory]. An argument that is not a device array, or
@racket[resources] that were released, raises raft's @racket[exn:fail:raft]
naming the procedure, and a number argument that does not fit the native type,
such as @racket[#:n-init 2.5], raises @racket[exn:fail:contract].

@(close-eval ev)
