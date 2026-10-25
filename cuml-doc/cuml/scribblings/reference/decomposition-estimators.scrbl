#lang scribble/manual
@(require scribble/decode
          scribble/example
          (for-label cuml
                     racket/base
                     racket/contract/base
                     raft
                     threading))

@(provide decomposition-estimators)

@(define (decomposition-estimators)
   (define ev (make-base-eval #:lang 'racket/base '(require cuml raft threading)))
   @splice[@list{
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
(~> (pca #:n-components 1) (fit P) (transform P) device-matrix->list*)
(eval:error (transform (pca) P))
]}

@defproc[(pca? [v any/c]) boolean?]{

Whether @racket[v] is a PCA estimator.}

@defproc[(pca-model [estimator pca?]) (or/c #f pca-result?)]{

The @racket[pca-result] of the fit that made @racket[estimator], or
@racket[#f] when it is unfitted.}

@section[#:tag "truncated-svd-estimator"]{Truncated SVD estimator}

@declare-exporting[cuml]

@defproc[(truncated-svd [params (or/c #f hash?) #f]
                        [#:n-components n-components exact-positive-integer? 1]
                        [#:solver solver (or/c 'full 'jacobi) 'full]
                        [#:tol tol (and/c real? (>=/c 0)) 0.0]
                        [#:iterations iterations (integer-in 1 2147483647) 15]
                        [#:resources resources (or/c #f device-resources?) #f])
         truncated-svd?]{

An unfitted truncated SVD with the hyperparameters
@racket[(tsvd-params params ...)]: an estimator and a transformer.
@racket[fit] is @racket[tsvd-fit]; @racket[transform] and
@racket[inverse-transform] are the procedures of the same name on the fit.
@racket[y] is ignored and a @racket[sample-weight] refused. Errors name the generic.

@examples[#:eval ev
(define S (list*->device-matrix '((1 2 0) (2 4 1) (3 6 0) (4 8 1)) #:dtype 'float64))
(device-matrix->list* (transform (fit (truncated-svd #:n-components 2) S) S))
(eval:error (transform (truncated-svd) S))
]}

@defproc[(truncated-svd? [v any/c]) boolean?]{

Whether @racket[v] is a truncated SVD estimator.}

@defproc[(truncated-svd-model [estimator truncated-svd?]) (or/c #f tsvd-result?)]{

The @racket[tsvd-result] of the fit that made @racket[estimator], or
@racket[#f] when it is unfitted.}

@(close-eval ev)}]))
