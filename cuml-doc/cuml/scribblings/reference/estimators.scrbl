#lang scribble/manual
@(require scribble/example
          (for-label cuml
                     racket/base
                     racket/contract/base
                     racket/generic
                     raft))

@(define host-ev (make-base-eval #:lang 'racket/base '(require cuml/estimator)))
@(define ev (make-base-eval #:lang 'racket/base '(require cuml raft)))

@title[#:tag "estimators"]{Estimators}

@section[#:tag "estimator-interface"]{The estimator interface}

@defmodule[cuml/estimator #:no-declare]
@declare-exporting[cuml/estimator cuml]

An estimator is a value that holds its parameters and, once fitted, what it
learned. The interface is a set of generic interfaces from
@racketmodname[racket/generic], one per capability; an estimator implements
@racket[gen:estimator] and whichever of the others it supports. Requiring
@racketmodname[cuml/estimator] loads no native library, so a host-side
library can implement it.

An implementation follows these rules, which nothing enforces:
@itemlist[
 @item{@racket[fit] returns a new estimator for which @racket[fitted?] is
       true, and leaves its argument unchanged;}
 @item{the methods that take @racket[y] and @racket[sample-weight] default
       both to @racket[#f];}
 @item{@racket[estimator-params] returns an immutable @racket[hasheq] of its
       hyperparameters, keyed by name;}
 @item{@racket[estimator-set-params] returns a new, unfitted estimator.}]

@defthing[gen:estimator any/c]{

The interface of every estimator, with the methods @racket[fit],
@racket[fitted?], @racket[estimator-params] and
@racket[estimator-set-params], all required.}

@defthing[gen:predictor any/c]{

An estimator that labels new samples: the method @racket[predict],
required.}

@defthing[gen:clusterer any/c]{

An estimator that clusters the samples it is fitted on: the method
@racket[fit-predict]. Its fallback, for an estimator that is also a
@racket[predictor?], is @racket[predict] after @racket[fit]; for any other
it raises @racket[exn:fail:support] without fitting.}

@defthing[gen:transformer any/c]{

An estimator that maps samples to new features: the methods
@racket[transform], required; @racket[fit-transform], whose fallback is
@racket[transform] after @racket[fit]; and @racket[inverse-transform],
optional; its fallback raises @racket[exn:fail:support].}

@defthing[gen:scorer any/c]{

An estimator that scores samples: the method @racket[score], required.}

@deftogether[(@defproc[(estimator? [v any/c]) boolean?]
              @defproc[(predictor? [v any/c]) boolean?]
              @defproc[(clusterer? [v any/c]) boolean?]
              @defproc[(transformer? [v any/c]) boolean?]
              @defproc[(scorer? [v any/c]) boolean?])]{

Whether @racket[v] implements @racket[gen:estimator], @racket[gen:predictor],
@racket[gen:clusterer], @racket[gen:transformer] or @racket[gen:scorer].}

@defproc[(fit [estimator estimator?]
              [X any/c]
              [y any/c #f]
              [#:sample-weight sample-weight any/c #f])
         estimator?]{

A fitted copy of @racket[estimator], fitted on the samples @racket[X], the
targets @racket[y] and the weights @racket[sample-weight]. Fitting a fitted
estimator fits again from its parameters.}

@defproc[(fitted? [estimator estimator?]) boolean?]{

Whether @racket[estimator] was returned by a @racket[fit].}

@defproc[(estimator-params [estimator estimator?]) (and/c hash? hash-eq? immutable?)]{

The parameters of @racket[estimator]: its hyperparameters, keyed by name.}

@defproc[(estimator-set-params [estimator estimator?] [changes hash? (hasheq)]) estimator?]{

An unfitted estimator with the parameters of @racket[estimator] updated by
@racket[changes], a hash of some of the keys of
@racket[(estimator-params estimator)]. With an empty @racket[changes] it is an
unfitted copy. It also takes each parameter as a keyword, which wins over the same key in
@racket[changes].}

@defproc[(predict [predictor predictor?] [X any/c]) any/c]{

The labels of the samples @racket[X].}

@defproc[(fit-predict [clusterer clusterer?]
                      [X any/c]
                      [y any/c #f]
                      [#:sample-weight sample-weight any/c #f])
         any/c]{

The labels of the samples @racket[X] after fitting @racket[clusterer] on
them.}

@defproc[(transform [transformer transformer?] [X any/c]) any/c]{

The samples @racket[X] mapped to the features @racket[transformer] learned.}

@defproc[(fit-transform [transformer transformer?]
                        [X any/c]
                        [y any/c #f]
                        [#:sample-weight sample-weight any/c #f])
         any/c]{

The samples @racket[X] transformed after fitting @racket[transformer] on
them.}

@defproc[(inverse-transform [transformer transformer?] [X any/c]) any/c]{

The samples whose @racket[transform] is @racket[X], for a transformer
that can undo its mapping.}

@defproc[(score [scorer scorer?]
                [X any/c]
                [y any/c #f]
                [#:sample-weight sample-weight any/c #f])
         real?]{

How well @racket[scorer] fits @racket[X], larger being better.}

An estimator on host lists that splits numbers at their mean, a predictor
and, through the fallback, a clusterer:

@examples[#:eval host-ev
(struct mean-split (cut)
  #:methods gen:estimator
  [(define (fit e X [y #f] #:sample-weight [w #f])
     (mean-split (/ (apply + X) (length X))))
   (define (fitted? e)
     (and (mean-split-cut e) #t))
   (define (estimator-params e)
     (hasheq))
   (define (estimator-set-params e params)
     (mean-split #f))]
  #:methods gen:predictor
  [(define (predict e X)
     (for/list ([x (in-list X)])
       (if (< x (mean-split-cut e)) 0 1)))]
  #:methods gen:clusterer
  [])
(fit-predict (mean-split #f) '(1 2 9 10))
(fitted? (mean-split #f))
]

@section[#:tag "kmeans-estimator"]{k-means estimator}

@declare-exporting[cuml]

@defproc[(kmeans [params (or/c #f hash?) #f]
                 [#:init init
                  (or/c 'scalable-k-means++ 'k-means++ 'random device-matrix?)
                  'scalable-k-means++]
                 [#:n-clusters n-clusters exact-positive-integer?
                  (if (device-matrix? init) (car (shape init)) 8)]
                 [#:max-iter max-iter exact-nonnegative-integer? 300]
                 [#:tol tol (and/c real? (>/c 0)) 1e-4]
                 [#:n-init n-init (or/c 'auto exact-positive-integer?) 'auto]
                 [#:random-state random-state exact-nonnegative-integer? 0]
                 [#:oversampling-factor oversampling-factor (and/c real? (>=/c 0)) 2.0]
                 [#:max-samples-per-batch max-samples-per-batch
                  (or/c #f (integer-in 0 2147483647))
                  #f]
                 [#:resources resources (or/c #f device-resources?) #f])
         kmeans?]{

An unfitted estimator with the hyperparameters @racket[(kmeans-params params ...)].
@racket[max-samples-per-batch] and @racket[resources] are settings, not hyperparameters:
@racket[#f] reads @racket[current-batch-samples] and @racket[(current-device-resources)] at each
call. @racket[fit] and @racket[estimator-set-params] keep them, @racket[equal?] compares them and
the printer shows them when given, but @racket[estimator-params] omits them. It is an estimator,
predictor, clusterer, transformer and scorer, and checks nothing until it is fitted.
@racket[n-clusters] is resolved here: changing @racket[init] later with
@racket[estimator-set-params] keeps it.

@racket[fit] is @racket[kmeans-fit] with these parameters and the
@racket[sample-weight]; @racket[predict], @racket[transform] and
@racket[score] are @racket[kmeans-predict], @racket[kmeans-transform] and
@racket[kmeans-score] on the fitted centroids; @racket[fit-predict] returns
the fit's labels. @racket[y] is ignored. Errors are @racket[kmeans-fit]'s, named
after the generic that was called; @racket[predict], @racket[transform] and
@racket[score] on an unfitted estimator raise @racket[exn:fail:cuml] of kind
@racket['logic].

An estimator prints the parameters that differ from the defaults, and two
are @racket[equal?] when their parameters are and they hold the same fit or
none.

@examples[#:eval ev
(define X
  (list*->device-matrix '((0 0) (0 2) (2 0) (2 2) (10 10) (10 12) (12 10) (12 12))
                        #:dtype 'float64))
(define km (kmeans #:n-clusters 2 #:random-state 1))
km
(define fitted (fit km X))
(list (fitted? km) (fitted? fitted))
(device-vector->list (predict fitted (list*->device-matrix '((1 1) (11 12)) #:dtype 'float64)))
(score fitted X)
(hash-ref (estimator-params km) 'n-clusters)
(fitted? (estimator-set-params fitted (hasheq 'max-iter 10)))
(estimator-set-params km #:max-iter 10)
(eval:error (predict km X))
]}

@defproc[(kmeans? [v any/c]) boolean?]{

Whether @racket[v] is a k-means estimator.}

@defproc[(kmeans-model [estimator kmeans?]) (or/c #f kmeans-result?)]{

The @racket[kmeans-result] of the fit that made @racket[estimator], or
@racket[#f] when it is unfitted.}

@section[#:tag "dbscan-estimator"]{DBSCAN estimator}

@declare-exporting[cuml]

@defproc[(dbscan [params (or/c #f hash?) #f]
                 [#:eps eps (and/c real? (>/c 0) (</c +inf.0)) 0.5]
                 [#:min-samples min-samples exact-positive-integer? 5]
                 [#:metric metric (or/c 'euclidean 'cosine 'precomputed) 'euclidean]
                 [#:out-dtype out-dtype (or/c 'int32 'int64) 'int32]
                 [#:max-mbytes-per-batch max-mbytes-per-batch
                  (or/c #f exact-nonnegative-integer?) #f]
                 [#:resources resources (or/c #f device-resources?) #f])
         dbscan?]{

An unfitted DBSCAN estimator with the hyperparameters @racket[(dbscan-params params ...)]. It
is an estimator and a clusterer, and checks nothing until it is fitted.
@racket[fit] is @racket[dbscan-fit] with these parameters and the
@racket[sample-weight], and @racket[fit-predict] returns the fit's labels.
@racket[y] is ignored. @racket[resources] is a setting, as for @racket[kmeans]: each call runs on
it, or with @racket[#f] on the calling thread's @racket[(current-device-resources)].
Errors are @racket[dbscan-fit]'s, named after the generic that was called.
Printing and @racket[equal?] are as for @racket[kmeans].

@examples[#:eval ev
(define P (list*->device-matrix '((0) (1) (2) (4) (6) (7) (8) (20)) #:dtype 'float64))
(define db (dbscan #:eps 2 #:min-samples 4))
db
(device-vector->list (fit-predict db P))
(estimator-set-params db #:metric 'cosine)
(eval:error (fit (dbscan #:eps 0) P))
]}

@defproc[(dbscan? [v any/c]) boolean?]{

Whether @racket[v] is a DBSCAN estimator.}

@defproc[(dbscan-model [estimator dbscan?]) (or/c #f dbscan-result?)]{

The @racket[dbscan-result] of the fit that made @racket[estimator], or
@racket[#f] when it is unfitted.}

@section[#:tag "agglomerative-estimator"]{Agglomerative clustering estimator}

@declare-exporting[cuml]

@defproc[(agglomerative-clustering
          [params (or/c #f hash?) #f]
          [#:n-clusters n-clusters exact-positive-integer? 2]
          [#:metric metric 'euclidean 'euclidean]
          [#:linkage linkage 'single 'single]
          [#:connectivity connectivity (or/c 'knn 'pairwise) 'knn]
          [#:c c (integer-in -2147483648 2147483617) 15]
          [#:resources resources (or/c #f device-resources?) #f])
         agglomerative-clustering?]{

An unfitted agglomerative clustering estimator with the hyperparameters
@racket[(single-linkage-params params ...)], whose @racket[linkage] is only
@racket['single]. It is an estimator and a clusterer, and checks nothing
until it is fitted. @racket[fit] is @racket[single-linkage-fit] with these
parameters, and @racket[fit-predict] returns the fit's labels. @racket[y] is
ignored, and a @racket[sample-weight] is refused. @racket[resources] is a
setting, as for @racket[kmeans]: each call runs on it, or with @racket[#f] on
the calling thread's @racket[(current-device-resources)]. Errors are
@racket[single-linkage-fit]'s, named after the generic that was called.
Printing and @racket[equal?] are as for @racket[kmeans].

@examples[#:eval ev
(define P (list*->device-matrix '((0) (1) (3) (7) (15)) #:dtype 'float32))
(define ac (agglomerative-clustering #:n-clusters 3))
ac
(device-vector->list (fit-predict ac P))
(estimator-set-params ac #:connectivity 'pairwise)
(eval:error (fit (agglomerative-clustering #:linkage 'ward) P))
]}

@defproc[(agglomerative-clustering? [v any/c]) boolean?]{

Whether @racket[v] is an agglomerative clustering estimator.}

@defproc[(agglomerative-clustering-model [estimator agglomerative-clustering?])
         (or/c #f single-linkage-result?)]{

The @racket[single-linkage-result] of the fit that made @racket[estimator], or
@racket[#f] when it is unfitted.}

@section[#:tag "spectral-estimator"]{Spectral clustering estimator}

@declare-exporting[cuml]

@defproc[(spectral-clustering
          [params (or/c #f hash?) #f]
          [#:n-clusters n-clusters exact-positive-integer? 8]
          [#:n-components n-components (or/c #f exact-positive-integer?) #f]
          [#:random-state random-state (integer-in 0 18446744073709551615) 0]
          [#:n-neighbors n-neighbors (and/c exact-integer? (>=/c 2)) 10]
          [#:n-init n-init (integer-in 1 2147483647) 10]
          [#:eigen-tol eigen-tol (or/c 'auto (and/c real? (>=/c 0))) 'auto]
          [#:affinity affinity 'nearest-neighbors 'nearest-neighbors]
          [#:resources resources (or/c #f device-resources?) #f])
         spectral-clustering?]{

An unfitted spectral clustering estimator with the hyperparameters
@racket[(spectral-clustering-params params ...)], whose @racket[affinity] is
only @racket['nearest-neighbors]. It is an estimator and a clusterer, and
checks nothing until it is fitted. @racket[fit] is
@racket[spectral-clustering-fit] with these parameters, and
@racket[fit-predict] returns the fit's labels. @racket[y] is ignored, and a
@racket[sample-weight] is refused. @racket[resources] is a setting, as for
@racket[kmeans]: each call runs on it, or with @racket[#f] on the calling
thread's @racket[(current-device-resources)]. Errors are
@racket[spectral-clustering-fit]'s, named after the generic that was called.
Printing and @racket[equal?] are as for @racket[kmeans].

@examples[#:eval ev
(define S
  (list*->device-matrix '((0 0) (0 1) (0 2) (1 0) (1 1) (1 2)
                          (10 0) (10 1) (10 2) (11 0) (11 1) (11 2))
                        #:dtype 'float32))
(define sc (spectral-clustering #:n-clusters 2 #:n-neighbors 4))
sc
(device-vector->list (fit-predict sc S))
(estimator-set-params sc #:n-init 3)
(eval:error (fit (spectral-clustering #:affinity 'precomputed) S))
]}

@defproc[(spectral-clustering? [v any/c]) boolean?]{

Whether @racket[v] is a spectral clustering estimator.}

@defproc[(spectral-clustering-model [estimator spectral-clustering?])
         (or/c #f spectral-clustering-result?)]{

The @racket[spectral-clustering-result] of the fit that made
@racket[estimator], or @racket[#f] when it is unfitted.}

@section[#:tag "hdbscan-estimator"]{HDBSCAN estimator}

@declare-exporting[cuml]

@defproc[(hdbscan [params (or/c #f hash?) #f]
                  [#:min-cluster-size min-cluster-size (integer-in 2 2147483647) 5]
                  [#:min-samples min-samples (or/c #f (integer-in 1 1023)) #f]
                  [#:cluster-selection-epsilon cluster-selection-epsilon (and/c real? (>=/c 0)) 0.0]
                  [#:max-cluster-size max-cluster-size (integer-in 0 2147483647) 0]
                  [#:metric metric 'euclidean 'euclidean]
                  [#:alpha alpha (and/c real? (>/c 0)) 1.0]
                  [#:cluster-selection-method cluster-selection-method (or/c 'eom 'leaf) 'eom]
                  [#:allow-single-cluster allow-single-cluster boolean? #f]
                  [#:build-algo build-algo 'brute-force 'brute-force]
                  [#:resources resources (or/c #f device-resources?) #f])
         hdbscan?]{

An unfitted HDBSCAN estimator with the hyperparameters
@racket[(hdbscan-params params ...)]. It is an estimator and a clusterer, not
a predictor, and checks nothing until it is fitted. @racket[fit] is
@racket[hdbscan-fit] with these parameters, and @racket[fit-predict] returns
the fit's labels. @racket[y] is ignored, and a @racket[sample-weight] is
refused. @racket[resources] is a setting, as for @racket[kmeans]. Errors are
@racket[hdbscan-fit]'s, named after the generic that was called, so a
@racket[min-cluster-size] above 1023 needs an explicit @racket[min-samples].
Printing and
@racket[equal?] are as for @racket[kmeans]. A refit leaves the earlier fit's
@tech{native model} alone.

@examples[#:eval ev
(define H
  (list*->device-matrix (for*/list ([centre (in-list '(0 20))]
                                    [i (in-range 10)])
                          (list (+ centre (quotient i 5)) (remainder i 5)))
                        #:dtype 'float32))
(define hd (hdbscan #:min-cluster-size 4))
hd
(device-vector->list (fit-predict hd H))
(estimator-set-params hd #:cluster-selection-method 'leaf)
(eval:error (fit (hdbscan #:alpha 0) H))
(eval:error (fit (hdbscan #:min-cluster-size 2000) H))
]}

@defproc[(hdbscan? [v any/c]) boolean?]{

Whether @racket[v] is an HDBSCAN estimator.}

@defproc[(hdbscan-model [estimator hdbscan?]) (or/c #f hdbscan-result?)]{

The @racket[hdbscan-result] of the fit that made @racket[estimator], or
@racket[#f] when it is unfitted.}

@section[#:tag "standard-scaler-estimator"]{Standard scaler estimator}

@declare-exporting[cuml]

@defproc[(standard-scaler [params (or/c #f hash?) #f]
                          [#:with-mean with-mean boolean? #t]
                          [#:with-std with-std boolean? #t]
                          [#:resources resources (or/c #f device-resources?) #f])
         standard-scaler?]{

An unfitted standard scaler with the hyperparameters
@racket[(standard-scaler-params params ...)]: an estimator and a transformer.
@racket[fit] is @racket[standard-scaler-fit]; @racket[transform],
@racket[inverse-transform] and @racket[fit-transform] are the procedures of
the same name on the fit. @racket[y] is ignored and a @racket[sample-weight]
refused. Errors name the generic.

@examples[#:eval ev
(define S (list*->device-matrix '((1 10) (3 10) (1 14) (3 14)) #:dtype 'float64))
(define s (fit (standard-scaler) S))
(device-matrix->list* (transform s S))
(eval:error (transform (standard-scaler) S))
]}

@defproc[(standard-scaler? [v any/c]) boolean?]{

Whether @racket[v] is a standard scaler estimator.}

@defproc[(standard-scaler-model [estimator standard-scaler?]) (or/c #f standard-scaler-result?)]{

The @racket[standard-scaler-result] of the fit that made @racket[estimator],
or @racket[#f] when it is unfitted.}

@section[#:tag "pca-estimator"]{PCA estimator}

@declare-exporting[cuml]

@defproc[(pca [params (or/c #f hash?) #f]
              [#:n-components n-components (or/c #f exact-positive-integer?) #f]
              [#:whiten whiten boolean? #f]
              [#:solver solver (or/c 'full 'jacobi) 'full]
              [#:tol tol (and/c real? (>=/c 0)) 0.0]
              [#:iterations iterations (integer-in 1 2147483647) 15]
              [#:resources resources (or/c #f device-resources?) #f])
         pca?]{

An unfitted principal component analysis with the hyperparameters
@racket[(pca-params params ...)]: an estimator and a transformer.
@racket[fit] is @racket[pca-fit]; @racket[transform] and
@racket[inverse-transform] are the procedures of the same name on the fit.
@racket[y] is ignored and a @racket[sample-weight] refused. Errors name the generic.

@examples[#:eval ev
(define P (list*->device-matrix '((4 0) (-4 0) (0 1) (0 -1) (0 0)) #:dtype 'float64))
(device-matrix->list* (transform (fit (pca #:n-components 1) P) P))
(eval:error (transform (pca) P))
]}

@defproc[(pca? [v any/c]) boolean?]{

Whether @racket[v] is a PCA estimator.}

@defproc[(pca-model [estimator pca?]) (or/c #f pca-result?)]{

The @racket[pca-result] of the fit that made @racket[estimator], or
@racket[#f] when it is unfitted.}

@include-section["tsvd-estimator.scrbl"]

@(close-eval host-ev)
@(close-eval ev)
