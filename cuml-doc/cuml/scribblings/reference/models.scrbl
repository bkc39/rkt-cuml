#lang scribble/manual
@(require scribble/example
          (for-label cuml
                     racket/base
                     racket/contract/base
                     raft))

@(define ev (make-base-eval #:lang 'racket/base '(require cuml raft)))

@title[#:tag "native-models"]{Native models}

@defmodule[cuml/core #:no-declare]
@declare-exporting[cuml/core cuml]

A @tech{native model} holds the C++ state behind a fitted result, such as
HDBSCAN's condensed tree and prediction data.

@examples[#:eval ev
(define X
  (list*->device-matrix (for*/list ([centre (in-list '(0 20))]
                                    [i (in-range 10)])
                          (list (+ centre (quotient i 5)) (remainder i 5)))
                        #:dtype 'float32))
(define result (hdbscan-fit X))
(define m (hdbscan-result-model result))
m
(native-model-released? m)
]

@defproc[(native-model? [v any/c]) boolean?]{

Whether @racket[v] is a native model. A native model keeps its device memory
alive on its own. Releasing it frees that memory now; otherwise it is freed
once the model and the arrays taken from its result are collected.}

@defproc[(native-model-release! [m native-model?]
                                [#:resources resources device-resources? (current-device-resources)])
         void?]{

Frees @racket[m]'s device memory now, on @racket[resources]. Calling it again
does nothing. Arrays already taken from @racket[m]'s result stay valid, but
using @racket[m] afterwards raises @racket[exn:fail:cuml].

@examples[#:eval ev
(native-model-release! m)
(native-model-released? m)
(device-vector->list (hdbscan-result-labels result))
(eval:error (hdbscan-approximate-predict result X))
]}

@defproc[(native-model-released? [m native-model?]) boolean?]{

Whether @racket[m] has been released.}

@defform[(with-native-models ([id model-expr] ...) body ...+)
         #:contracts ([model-expr native-model?])]{

Binds each @racket[id] to its model, checking every one before holding any,
and releases each when @racket[body] exits by any route, a raise included.

@examples[#:eval ev
(define held (hdbscan-result-model (hdbscan-fit X)))
(with-native-models ([h held])
  (native-model-released? h))
(native-model-released? held)
]}

@(close-eval ev)
