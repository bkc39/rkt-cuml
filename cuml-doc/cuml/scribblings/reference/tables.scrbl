#lang scribble/manual
@(require scribble/example
          (for-label cuml
                     datasets/core
                     racket/base
                     racket/contract/base
                     racket/list
                     raft))

@(define ev (make-base-eval #:lang 'racket/base '(require cuml datasets/core racket/list raft)))

@title[#:tag "tables"]{Tables}

@declare-exporting[cuml/table cuml]

@defproc[(table->device-matrix [table dataset-table?]
                               [#:columns columns (or/c #f (listof string?)) #f]
                               [#:dtype dtype (or/c 'float32 'float64) 'float32]
                               [#:layout layout (or/c 'row-major 'col-major) 'row-major]
                               [#:resources resources device-resources? (current-device-resources)])
         device-matrix?]{

Copies the columns of @racket[table] named by @racket[columns], in that order,
to a new @tech{sample matrix} on @racket[resources]: one row per table row, one
matrix column per selected column. A @racket[columns] of @racket[#f] selects
every column in the table's order. Inexact values are rounded to the nearest
@racket[dtype] value, and NaN and infinities are copied. A missing value, a
value that is not a real number, an exact number that @racket[dtype] cannot
hold exactly, or a finite value beyond @racket[dtype]'s range raises
@racket[exn:fail:cuml] naming the column and the row, before any device memory
is allocated.

@examples[#:eval ev
(define anscombe (load-dataset-table 'anscombe))
(define X (table->device-matrix anscombe #:columns '("x1" "y1") #:dtype 'float64))
(shape X)
(take (device-matrix->list* X) 3)
(eval:error (table->device-matrix (load-dataset-table 'iris)))
(eval:error (table->device-matrix (load-dataset-table 'airquality) #:columns '("ozone")))
]}
