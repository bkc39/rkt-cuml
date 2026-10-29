#lang scribble/manual
@(require "../../utils.rkt")

@(define ev (make-cuml-eval))

@title[#:tag "ex-hdbscan"]{HDBSCAN: Old Faithful without a radius}

HDBSCAN builds DBSCAN's clusters at every radius at once, as a tree, and keeps
the clusters that persist longest. It needs no @racket[eps]: only
@racket[min-cluster-size], the smallest group worth calling a cluster, and
@racket[min-samples], which sets how dense a neighbourhood must be. It suits
groups of different densities, which no single DBSCAN radius fits.

@section[#:tag "ex-hdbscan-fit"]{Two groups and noise}

The eruptions are those of @secref["ex-dbscan"], standardized:

@examples[#:eval ev #:label #f
(define-values (faithful-scaler eruptions)
  (~> (load-dataset-table 'faithful)
      (table->device-matrix #:columns '("eruptions" "waiting"))
      standard-scaler-fit-transform))
(define labels (labels-of (hdbscan) eruptions))
(list (sizes labels) (count negative? labels))
]

At its defaults HDBSCAN finds the two groups and 4 eruptions of noise, where
DBSCAN needed a radius chosen between 0.2 and 0.5. Beside DBSCAN's fit at 0.3:

@examples[#:eval ev #:label #f
(define lengths (dataset-column 'faithful "eruptions"))
(define waits (dataset-column 'faithful "waiting"))
(define dbscan-labels (labels-of (dbscan #:eps 0.3 #:min-samples 5) eruptions))
(sweep-figure lengths waits
              '("DBSCAN, eps 0.3" "HDBSCAN")
              (list dbscan-labels (matched-labels labels dbscan-labels))
              #:x-label "eruption (min)"
              #:y-label "wait (min)")
]

@section[#:tag "ex-hdbscan-predict"]{New eruptions}

A fit keeps what @racket[hdbscan-approximate-predict] needs to place new
samples without refitting. Fit the first 200 eruptions and place the last 72:

@examples[#:eval ev #:label #f
(define scaled (device-matrix->list* eruptions))
(define first-200 (hdbscan-fit (rows->matrix (take scaled 200))))
(define placed
  (~> (hdbscan-approximate-predict first-200 (rows->matrix (drop scaled 200)))
      hdbscan-prediction-labels
      device-vector->list))
(list (sizes placed) (count negative? placed))
(cluster-plot (drop lengths 200)
              (drop waits 200)
              placed
              #:title "the last 72, placed"
              #:x-label "eruption (min)"
              #:y-label "wait (min)")
]

The fit holds C++ memory for these predictions, a @tech{native model},
freed when the result is collected or at once with
@racket[native-model-release!].

@(close-eval ev)
