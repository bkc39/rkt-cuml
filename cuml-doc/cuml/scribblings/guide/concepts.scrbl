#lang scribble/manual
@(require "../utils.rkt")

@(define ev (make-cuml-eval))

@title[#:tag "concepts" #:style 'toc]{Concepts}

Every algorithm in @racketmodname[cuml] takes its data the same way, comes in
two forms, and refuses bad arguments the same way. This chapter covers those
shared pieces on the iris data of @secref["getting-started"].

@local-table-of-contents[]

The data comes from the helpers of @secref["examples"], and
@racket[kmeans-labels] copies a k-means result's labels to the host:

@examples[#:eval ev #:label #f
(define measurements (features 'iris "species"))
(define flowers (dataset-matrix 'iris measurements))
(define species (dataset-column 'iris "species"))
(define (kmeans-labels result)
  (device-vector->list (kmeans-result-labels result)))
]

@section[#:tag "concepts-data"]{Device data}

A @deftech{sample matrix} is a raft @racket[device-matrix], row-major, one
sample per row and one feature per column. k-means and DBSCAN take
@racket['float32] or @racket['float64]; single linkage and spectral clustering
take only @racket['float32]:

@examples[#:eval ev #:label #f
(define flowers64 (dataset-matrix 'iris measurements #:dtype 'float64))
(dtype flowers64)
(~> (kmeans-fit flowers64 #:n-clusters 3) kmeans-labels (crosstab species))
(eval:error (single-linkage-fit flowers64 #:n-clusters 3))
]

A column-major matrix is refused, not transposed:

@examples[#:eval ev #:label #f
(eval:error
 (~> (dataset-columns 'iris measurements)
     rows
     (list*->device-matrix #:dtype 'float32 #:layout 'col-major)
     (kmeans-fit #:n-clusters 3)))
]

Results are device arrays too; @racket[device-vector->list] and
@racket[device-matrix->list*] copy them to the host.

@section[#:tag "concepts-resources"]{Resources}

Every call runs on raft @deftech{device resources}: a device, a stream and the
library handles. A procedure takes them as @racket[#:resources], by default
the calling thread's @racket[(current-device-resources)], and its results live
there. An estimator's @racket[#:resources] of @racket[#f] means the current
thread's at each call. @racket[with-device-resources] scopes a fresh set:

@examples[#:eval ev #:label #f
(with-device-resources ([r (device-resources)])
  (~> (kmeans-fit flowers #:n-clusters 3 #:resources r)
      kmeans-labels
      (crosstab species)
      agreement))
]

@section[#:tag "concepts-estimators"]{Procedures and estimators}

Each algorithm comes in two forms. A @deftech{fit procedure}, such as
@racket[kmeans-fit], takes the data and the parameters and returns a result
struct. An @deftech{estimator}, such as @racket[kmeans], holds the parameters
until @racket[fit] returns a fitted copy, in the style of scikit-learn;
@racket[fit-predict] fits and returns the labels:

@examples[#:eval ev #:label #f
(define estimator (kmeans #:n-clusters 3))
(equal? (device-vector->list (fit-predict estimator flowers))
        (kmeans-labels (kmeans-fit flowers #:n-clusters 3)))
(hash-ref (estimator-params estimator) 'n-clusters)
(estimator-set-params estimator #:n-clusters 2)
(kmeans-model (fit estimator flowers))
]

An estimator checks nothing until it is fitted. @racket[kmeans-model] and its
siblings return the fit procedure's result.

@section[#:tag "concepts-parameters"]{Parameters}

A setting that changes the result is a @deftech{hyperparameter}. Each
algorithm keeps its hyperparameters in a set, an immutable hash that
@racket[kmeans-params] and its siblings build and check. A fit procedure takes a
set as @racket[#:params], an estimator as its first argument, and a keyword
wins over the same key in the set. A setting that changes only speed, memory or
the device is an @deftech{execution setting}, kept out of the set: a parameter
such as @racket[current-batch-samples], or a constructor setting such as
@racket[#:resources]:

@examples[#:eval ev #:label #f
(define base (kmeans-params #:n-init 3))
(for/list ([k (in-range 2 5)])
  (~> (fit (kmeans base #:n-clusters k) flowers) kmeans-model kmeans-labels (crosstab species) agreement))
(hash-ref (estimator-params (estimator-set-params (kmeans base) #:n-clusters 2)) 'n-init)
(define (base-labels)
  (kmeans-labels (kmeans-fit flowers #:params base #:n-clusters 3)))
(equal? (base-labels) (parameterize ([current-batch-samples 16]) (base-labels)))
]

@section[#:tag "concepts-labels"]{Labels}

The @deftech{labels} are an @racket['int32] device vector with one cluster
number per sample. The numbers name clusters, not classes, and their order is
the algorithm's: k-means numbers its clusters by its seeding, DBSCAN by each
cluster's smallest core sample with @racket[-1] for noise, single linkage by the
dendrogram, and spectral clustering by the k-means it runs on its embedding.
Two seeds can find the same partition under different numbers:

@examples[#:eval ev #:label #f
(define (labels-at seed)
  (kmeans-labels (kmeans-fit flowers #:n-clusters 3 #:random-state seed)))
(equal? (labels-at 0) (labels-at 1))
(equal? (crosstab (labels-at 0) species) (crosstab (labels-at 1) species))
]

Compare partitions, as @racket[crosstab] does, not label values.

@section[#:tag "concepts-refusals"]{Refusals}

A @deftech{refusal} is an @racket[exn:fail:cuml] of kind @racket['logic],
raised before the GPU is touched and leaving nothing allocated. Its message
names the procedure that was called, which for an estimator is the generic:

@examples[#:eval ev #:label #f
(eval:error (kmeans-fit flowers #:n-clusters 151))
(eval:error (fit (kmeans #:n-clusters 151) flowers))
]

A number that does not fit the native type is a contract error instead:

@examples[#:eval ev #:label #f
(eval:error (kmeans-fit flowers #:n-clusters 3 #:n-init 2.5))
]

Each procedure's entry in the @secref["reference"] lists its refusals.

@section[#:tag "concepts-native-models"]{Native models}

Most results are device arrays alone. An HDBSCAN result also holds a
@deftech{native model}: the C++ state its prediction procedures read, which
keeps its device memory until the model and every array taken from the result
are collected. @racket[native-model-release!] frees it at once, and
@racket[with-native-models] frees it when a block exits. The arrays stay
valid; only predicting from a released model is refused.

@section[#:tag "concepts-random-state"]{Reproducibility}

k-means seeds its centroids, and spectral clustering its k-means, from
@racket[#:random-state], 0 by default. The same seed gives the same labels;
another seed can renumber the clusters, or move a sample near a boundary:

@examples[#:eval ev #:label #f
(equal? (labels-at 2) (labels-at 2))
(crosstab (labels-at 2) species)
]

DBSCAN and single linkage take no seed, but are not always deterministic:
which cluster a DBSCAN border sample joins can depend on the batching, and
single linkage's @racket['knn] graph can order tied merges differently.

@(close-eval ev)
