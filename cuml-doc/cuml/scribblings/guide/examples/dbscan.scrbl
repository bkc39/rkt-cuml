#lang scribble/manual
@(require "../../utils.rkt")

@(define ev (make-cuml-eval))

@title[#:tag "ex-dbscan"]{DBSCAN: Old Faithful eruptions}

DBSCAN grows clusters from core samples, those with at least
@racket[min-samples] samples within @racket[eps]. Core samples within
@racket[eps] of each other share a cluster, samples near a core sample join
it, and the rest are noise. It finds the number of clusters itself and leaves stragglers out, so it suits
dense groups separated by sparse regions, at one density.

@section[#:tag "ex-dbscan-data"]{The data}

The faithful data records 272 eruptions of the Old Faithful geyser: each
eruption's length and the wait until the next, both in minutes. Short
eruptions are followed by short waits and long ones by long waits: two dense
groups, with a few eruptions between. The two columns differ in range, so
standardize them:

@examples[#:eval ev #:label #f
(define X
  (~> (load-dataset 'faithful #:format 'table)
      standardized-rows
      (list*->device-matrix #:dtype 'float32)))
(shape X)
]

@section[#:tag "ex-dbscan-fit"]{Two groups and noise}

@examples[#:eval ev #:label #f
(define labels (device-vector->list (fit-predict (dbscan #:eps 0.3 #:min-samples 5) X)))
(sizes (filter exact-nonnegative-integer? labels))
(count negative? labels)
]

DBSCAN finds the two groups without being told how many there are, and marks
8 eruptions between and around them as noise, which k-means would assign to
one group or the other.

@section[#:tag "ex-dbscan-eps"]{The radius}

@racket[eps] is a distance in the units of @racket[X], here standard
deviations. Too small, and the groups break up into fragments and noise; too
large, and the eruptions between the groups join them into one:

@examples[#:eval ev #:label #f
(for/list ([eps (in-list '(0.1 0.2 0.3 0.5))])
  (define labels (device-vector->list (fit-predict (dbscan #:eps eps #:min-samples 5) X)))
  (list eps (sizes (filter exact-nonnegative-integer? labels)) (count negative? labels)))
]

@racket[dbscan-fit] also returns the core samples:

@examples[#:eval ev #:label #f
(~> (dbscan-fit X #:eps 0.3 #:min-samples 5)
    dbscan-result-core-sample-indices
    device-vector->list
    length)
]

252 of the 272 eruptions are core samples, 8 are noise, and the other 12
join a cluster as border samples.

@(close-eval ev)
