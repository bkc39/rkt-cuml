#lang scribble/manual
@(require scribble/example
          (for-label cuml
                     racket/base
                     racket/contract/base
                     raft))

@(define ev (make-base-eval #:lang 'racket/base '(require cuml raft)))
@(define int32-count @racket[(integer-in 1 2147483647)])

@title[#:tag "svm"]{Support vector machines}

@defmodule[cuml/svm #:no-declare]
@declare-exporting[cuml/svm cuml]

@examples[#:eval ev #:hidden
(define X (list*->device-matrix '((0.0 0.0) (0.0 1.0) (1.0 0.0) (3.0 3.0) (3.0 4.0) (4.0 3.0))
                                #:dtype 'float64))
(define y '(a a a b b b))
]

@defstruct*[svc-result ([support-vectors device-matrix?]
                        [support device-vector?]
                        [dual-coef device-matrix?]
                        [intercept device-vector?]
                        [n-support (list/c exact-nonnegative-integer? exact-nonnegative-integer?)]
                        [classes list?]
                        [label-kind any/c]
                        [n-iter exact-nonnegative-integer?]
                        [kernel symbol?]
                        [degree exact-nonnegative-integer?]
                        [gamma flonum?]
                        [coef0 flonum?]
                        [cache-size flonum?])]{
A fitted binary C-SVC: the rows of X with nonzero dual coefficient, in row order; their indices;
y·α, positive for the second class; the intercept; and the count of support vectors in each class.
@racket[gamma] is the value the fit used. Unlike Python cuML's total, @racket[n-support] counts
each class, as scikit-learn's does. @racket[label-kind] is opaque, as
@racket[logistic-regression-result]'s.}

@defproc[(svc-fit [X device-matrix?]
                  [y (or/c device-vector? list?)]
                  [#:sample-weight sample-weight (or/c #f device-vector?) #f]
                  [#:kernel kernel (or/c 'linear 'poly 'rbf 'sigmoid) 'rbf]
                  [#:degree degree exact-nonnegative-integer? 3]
                  [#:gamma gamma (or/c 'scale 'auto (and/c rational? (>=/c 0))) 'scale]
                  [#:coef0 coef0 rational? 0.0]
                  [#:C C (and/c rational? positive?) 1.0]
                  [#:tol tol (and/c rational? positive?) 1e-3]
                  [#:cache-size cache-size (and/c rational? positive?) 1024.0]
                  [#:max-iter max-iter (or/c #f #,int32-count) #f]
                  [#:nochange-steps nochange-steps exact-nonnegative-integer? 1000]
                  [#:params params (or/c #f hash?) #f]
                  [#:resources resources device-resources? (current-device-resources)])
         svc-result?]{
Solves the C-SVC dual by SMO: minimise ½αᵀQα − Σα subject to 0 ≤ αᵢ ≤ C·wᵢ and Σyᵢαᵢ = 0,
Qᵢⱼ = yᵢyⱼk(xᵢ, xⱼ), until the largest violating pair is within @racket[tol]. @racket[y] holds
exactly two classes: integral numbers on the device, or a list of reals, strings or symbols.
@racket['scale] is 1/(d·Var X) over every element, @racket['auto] 1/d; @racket['sigmoid] is
tanh(γ⟨x, x′⟩ + coef0), @racket['poly] (γ⟨x, x′⟩ + coef0)^degree. @racket[max-iter] bounds the
pair updates (@racket[#f]: none). X is used as given: kernel SVMs are not scale-invariant, so
standardise it. @racket[cache-size] (MiB) is the kernel cache; the fit allocates all of it on the
device, so the default takes 1 GiB. A smaller cache can take another path to another fit within
@racket[tol] (up to 0.02 in the decision on 300×4 at 1 MiB). A cache whose index overflows int32 for
X's rows is refused, naming the largest that fits. scikit-learn's default is 200. The
sigmoid kernel is not positive semidefinite, so its fit is one stationary point among several.
The fit repeats bitwise.
@examples[#:eval ev #:label #f
(define r (svc-fit X y #:kernel 'linear))
(svc-result-n-support r)
(device-vector->list (svc-result-support r))
(eval:error (svc-fit X '(a a a a a a)))
(eval:error (svc-fit X y #:gamma -1.0))
]}

@defproc[(svc-decision-function [result svc-result?]
                                [X device-matrix?]
                                [#:resources resources device-resources? (current-device-resources)])
         device-vector?]{
Σᵢ dual-coefᵢ·k(svᵢ, x) + intercept for each row, in the fit's dtype.
@examples[#:eval ev #:label #f
(device-vector->list (svc-decision-function r X))
]}

@defproc[(svc-predict [result svc-result?]
                      [X device-matrix?]
                      [#:resources resources device-resources? (current-device-resources)])
         (or/c device-vector? list?)]{
The second class where the decision function is at least 0, else the first, as @racket[y] was
given.
@examples[#:eval ev #:label #f
(svc-predict r X)
]}

@defproc[(svc-params [base (or/c #f hash?) #f]
                     [#:kernel kernel (or/c 'linear 'poly 'rbf 'sigmoid) 'rbf]
                     [#:degree degree exact-nonnegative-integer? 3]
                     [#:gamma gamma (or/c 'scale 'auto (and/c rational? (>=/c 0))) 'scale]
                     [#:coef0 coef0 rational? 0.0]
                     [#:C C (and/c rational? positive?) 1.0]
                     [#:tol tol (and/c rational? positive?) 1e-3]
                     [#:cache-size cache-size (and/c rational? positive?) 1024.0]
                     [#:max-iter max-iter (or/c #f #,int32-count) #f]
                     [#:nochange-steps nochange-steps exact-nonnegative-integer? 1000])
         (and/c hash? hash-eq? immutable?)]{
The hyperparameters of @racket[svc-fit] and @racket[svc], as @racket[kmeans-params].}

@section[#:tag "svc-estimator"]{SVC estimator}

@declare-exporting[cuml]

@defproc[(svc [params (or/c #f hash?) #f]
              [#:kernel kernel (or/c 'linear 'poly 'rbf 'sigmoid) 'rbf]
              [#:degree degree exact-nonnegative-integer? 3]
              [#:gamma gamma (or/c 'scale 'auto (and/c rational? (>=/c 0))) 'scale]
              [#:coef0 coef0 rational? 0.0]
              [#:C C (and/c rational? positive?) 1.0]
              [#:tol tol (and/c rational? positive?) 1e-3]
              [#:cache-size cache-size (and/c rational? positive?) 1024.0]
              [#:max-iter max-iter (or/c #f #,int32-count) #f]
              [#:nochange-steps nochange-steps exact-nonnegative-integer? 1000]
              [#:resources resources (or/c #f device-resources?) #f])
         svc?]{
The estimator: @racket[fit] is @racket[svc-fit] with @racket[#:sample-weight] and needs
@racket[y]; @racket[predict] and @racket[decision-function] are the @racket[svc-] functions; and
@racket[score] is the weighted @racket[accuracy-score], labels outside @racket[classes] counting
as wrong.
@examples[#:eval ev #:label #f
(score (fit (svc #:kernel 'linear) X y) X y)
]}

@defproc[(svc? [v any/c]) boolean?]{Whether @racket[v] is an svc estimator.}

@defproc[(svc-model [estimator svc?]) (or/c #f svc-result?)]{
The @racket[svc-result] of the fit that made @racket[estimator], or @racket[#f].}

@(close-eval ev)
