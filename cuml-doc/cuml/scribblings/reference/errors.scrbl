#lang scribble/manual
@(require scribble/example
          (for-label cuml
                     racket/base
                     raft))

@(define ev (make-base-eval #:lang 'racket/base '(require cuml/core raft)))

@title[#:tag "errors"]{Errors}

@defmodule[#:multi (cuml cuml/core)]

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
