#lang scribble/manual
@(require scribble/example
          (for-label cuml
                     racket/base
                     raft))

@(define ev (make-base-eval #:lang 'racket/base '(require cuml raft)))

@title{cuML: GPU machine learning for Racket}
@author{bkc}

This library binds @hyperlink["https://github.com/rapidsai/cuml"]{NVIDIA
cuML}, the RAPIDS library of machine-learning algorithms on the GPU. It works
on the device arrays and device resources of @racketmodname[raft], and
reports failures as raft does.

@bold{License.} This package is distributed under @bold{Apache-2.0}.

@bold{Acknowledgements.} cuML and RAFT, which do all of this package's work,
are developed by NVIDIA's RAPIDS cuML and RAFT teams.

@bold{AI Disclosure.} This package and its documentation were created with the
use of AI tools.

Install it with @exec{raco pkg install cuml}.

@local-table-of-contents[]

@section[#:tag "errors"]{Errors}

@defmodule[cuml]

@defstruct*[(exn:fail:cuml exn:fail:raft) ()]{

Raised when cuML refuses or fails a call. The message starts with the Racket
procedure that was called, then cuML's own message, and the
@racket[exn:fail:raft-kind] field says what failed, as for
@racket[exn:fail:raft]. Since it is a subtype, a handler for
@racket[exn:fail:raft] also catches it:

@examples[#:eval ev
(with-handlers ([exn:fail:raft? exn:fail:raft-kind])
  (raise (exn:fail:cuml "my-binding: n-clusters: expected at least 1, got 0"
                        (current-continuation-marks)
                        'logic)))
]

The kind tells a retry apart from a mistake:

@examples[#:eval ev #:label #f
(define e
  (exn:fail:cuml "my-binding: out of memory" (current-continuation-marks)
                 'out-of-memory))
(list (exn:fail:cuml? e) (exn:fail:raft? e) (exn:fail:raft-kind e))
]

A failure inside raft itself is not a cuML failure:

@examples[#:eval ev #:label #f
(exn:fail:cuml?
 (exn:fail:raft "device-resources: no device 7" (current-continuation-marks)
                'logic))
]}

@(close-eval ev)
