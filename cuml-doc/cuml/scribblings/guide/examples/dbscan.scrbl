#lang scribble/manual
@(require "../../utils.rkt")

@(define ev (make-cuml-eval))

@title[#:tag "ex-dbscan"]{DBSCAN: Old Faithful eruptions}

DBSCAN grows clusters from core samples, those with at least
@racket[min-samples] samples within @racket[eps]. Core samples within
@racket[eps] of each other share a cluster, samples near a core sample join
it, and the rest are noise. It finds the number of clusters itself and leaves
stragglers out, so it suits dense groups separated by sparse regions, at one
density.

@section[#:tag "ex-dbscan-data"]{The data}

The faithful data (W. Härdle, 1991, from R 4.3.3's @tt{datasets} by the R Core
Team, under GPL-2 or GPL-3) records 272 eruptions of the Old Faithful geyser:
each eruption's length and the wait until the next, both in minutes. Short
eruptions are followed by short waits and long ones by long waits: two dense
groups, with a few eruptions between. The two columns differ in range, so
standardize them:

@examples[#:eval ev #:label #f
(define-values (faithful-scaler eruptions)
  (~> (load-dataset-table 'faithful)
      (table->device-matrix #:columns '("eruptions" "waiting"))
      standard-scaler-fit-transform))
(shape eruptions)
]

@section[#:tag "ex-dbscan-fit"]{Two groups and noise}

@racket[summary] lists the cluster sizes and counts the noise:

@examples[#:eval ev #:label #f
(define (summary eps)
  (define labels (labels-of (dbscan #:eps eps #:min-samples 5) eruptions))
  (list (sizes labels) (count negative? labels)))
(summary 0.3)
]

DBSCAN finds the two groups without being told how many there are, and marks
8 eruptions between and around them as noise, which k-means would assign to
one group or the other.

@section[#:tag "ex-dbscan-eps"]{The radius}

@racket[eps] is a distance in the units of the data, here standard
deviations. Too small, and the groups break up into fragments and noise; too
large, and the eruptions between the groups join them into one:

@examples[#:eval ev #:label #f
(for/list ([eps (in-list '(0.1 0.2 0.3 0.5))])
  (cons eps (summary eps)))
]

@racket[dbscan-fit] also returns the core samples:

@examples[#:eval ev #:label #f
(~> (dbscan-fit eruptions #:eps 0.3 #:min-samples 5)
    dbscan-result-core-sample-indices
    device-vector->list
    length)
]

252 of the 272 eruptions are core samples, 8 are noise, and the other 12
join a cluster as border samples.

As in scikit-learn's DBSCAN demo, core samples are filled, border samples
hollow and noise small gray circles. The eruptions are drawn in minutes, and
@racket[matched-labels] numbers each fit's clusters after those at 0.3, so a
group keeps its colour across the radii:

@examples[#:eval ev #:label #f
(define lengths (dataset-column 'faithful "eruptions"))
(define waits (dataset-column 'faithful "waiting"))
(define radii '(0.1 0.2 0.3 0.5))
(define fits
  (for/list ([eps (in-list radii)])
    (dbscan-fit eruptions #:eps eps #:min-samples 5)))
(define (labels-in result)
  (device-vector->list (dbscan-result-labels result)))
(define groups (labels-in (third fits)))
(sweep-figure lengths waits
              (map (curry format "eps ~a") radii)
              (map (lambda (result) (matched-labels (labels-in result) groups)) fits)
              #:cores (map (compose device-vector->list dbscan-result-core-sample-indices) fits)
              #:x-label "eruption (min)"
              #:y-label "wait (min)")
]

At 0.1, 126 eruptions are noise and the groups break into fragments; at 0.2
and 0.3 the two groups stand out; at 0.5 the eruptions between them join them
into one cluster.

@(close-eval ev)
