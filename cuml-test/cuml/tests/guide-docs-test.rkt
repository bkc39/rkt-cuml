#lang racket/base

(require (only-in cuml
                  agglomerative-clustering
                  dbscan
                  dbscan-fit
                  dbscan-result-core-sample-indices
                  estimator-params
                  estimator-set-params
                  fit
                  fit-predict
                  kmeans
                  kmeans-fit
                  kmeans-model
                  kmeans-result-labels
                  kmeans-result?
                  predict
                  score
                  single-linkage-fit
                  spectral-clustering)
         (only-in datasets/core
                  dataset-table-column
                  dataset-table-columns
                  dataset-table-names
                  load-dataset-table)
         (only-in datasets/matrix table->matrix)
         (only-in racket/list argmax count group-by index-of remove-duplicates)
         (only-in racket/math exact-round sqr)
         (only-in rackunit check-equal? check-exn check-not-equal? check-pred)
         (only-in raft
                  device-resources
                  device-vector->list
                  dtype
                  list*->device-matrix
                  shape
                  with-device-resources)
         (only-in raft/compat matrix->device-matrix)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/cuml-error.rkt" check-cuml-error))

(define (crosstab labels classes)
  (define names (remove-duplicates classes))
  (~> (for/list ([cluster (in-list (group-by car (map cons labels classes)))])
        (for/list ([name (in-list names)])
          (count (lambda (sample) (equal? (cdr sample) name)) cluster)))
      (sort < #:key (lambda (row) (index-of row (apply max row))))))

(define (agreement table)
  (for/sum ([row (in-list table)]) (apply max row)))

(define (sizes labels)
  (sort (map length (group-by values labels)) >))

(define (standardized-rows table)
  (~> (for/list ([column (in-vector (dataset-table-columns table))])
        (define n (vector-length column))
        (define mean (/ (for/sum ([v (in-vector column)]) v) n))
        (define sd (sqrt (/ (for/sum ([v (in-vector column)]) (sqr (- v mean))) n)))
        (for/list ([v (in-vector column)])
          (exact->inexact (/ (- v mean) sd))))
      (apply map list _)))

(define (matrix-of name columns)
  (table->matrix (load-dataset-table name #:columns columns)))

(define (scaled name columns)
  (~> (load-dataset-table name #:columns columns)
      standardized-rows
      (list*->device-matrix #:dtype 'float32)))

(define (column-of name column)
  (~> (load-dataset-table name) (dataset-table-column column) vector->list))

(define (labels-of estimator X)
  (device-vector->list (fit-predict estimator X)))

(define measurements '("sepal-length" "sepal-width" "petal-length" "petal-width"))

(define (iris)
  (matrix->device-matrix (matrix-of 'iris measurements) #:dtype 'float32))

(define species (column-of 'iris "species"))

(define iris-kmeans-table '((50 0 0) (0 48 14) (0 2 36)))

(test-gpu "the guide: iris on the device, and its three species"
  (check-equal? (shape (iris)) '(150 4))
  (check-equal? (remove-duplicates species) '("Iris-setosa" "Iris-versicolor" "Iris-virginica")))

(test-gpu "the guide: a first k-means against the species, and new flowers"
  (define fitted (fit (kmeans #:n-clusters 3) (iris)))
  (check-equal? (format "~v" fitted) "#<kmeans fitted #:n-clusters 3>")
  (define labels (~> fitted kmeans-model kmeans-result-labels device-vector->list))
  (check-equal? (length labels) 150)
  (check-equal? (crosstab labels species) iris-kmeans-table)
  (check-equal? (agreement (crosstab labels species)) 134)
  (define cluster-species
    (for/hash ([cluster (in-list (group-by car (map cons labels species)))])
      (values (caar cluster) (~> (map cdr cluster) (group-by values _) (argmax length _) car))))
  (define new-flowers
    (list*->device-matrix '((5.0 3.4 1.5 0.2) (5.9 2.8 4.3 1.3) (6.6 3.0 5.6 2.0)) #:dtype 'float32))
  (define new-labels (device-vector->list (predict fitted new-flowers)))
  (check-equal? (for/list ([label (in-list new-labels)])
                  (hash-ref cluster-species label))
                '("Iris-setosa" "Iris-versicolor" "Iris-virginica")))

(define (labels-at seed)
  (~> (kmeans-fit (iris) #:n-clusters 3 #:random-state seed)
      kmeans-result-labels
      device-vector->list))

(test-gpu "the guide: float64 for k-means, refused by single linkage, and col-major refused"
  (define X64 (matrix->device-matrix (matrix-of 'iris measurements)))
  (check-equal? (dtype X64) 'float64)
  (check-equal?
   (~> (kmeans-fit X64 #:n-clusters 3) kmeans-result-labels device-vector->list (crosstab species))
   iris-kmeans-table)
  (check-cuml-error 'logic
                    "single-linkage-fit: X: expected float32, got float64"
                    (lambda () (single-linkage-fit X64 #:n-clusters 3)))
  (check-cuml-error 'logic
                    "kmeans-fit: X: expected row-major, got col-major 150x4"
                    (lambda ()
                      (~> (matrix-of 'iris measurements)
                          (matrix->device-matrix #:dtype 'float32 #:layout 'col-major)
                          (kmeans-fit #:n-clusters 3)))))

(test-gpu "the guide: scoped resources, the two forms, and renumbering seeds"
  (check-equal? (with-device-resources ([r (device-resources)])
                  (~> (kmeans-fit (iris) #:n-clusters 3 #:resources r)
                      kmeans-result-labels
                      device-vector->list
                      (crosstab species)
                      agreement))
                134)
  (define km (kmeans #:n-clusters 3))
  (check-equal? (labels-of km (iris)) (labels-at 0))
  (check-equal? (hash-ref (estimator-params km) 'n-clusters) 3)
  (check-equal? (format "~v" (estimator-set-params km (hasheq 'n-clusters 2)))
                "#<kmeans #:n-clusters 2>")
  (check-pred kmeans-result? (kmeans-model (fit km (iris))))
  (check-not-equal? (labels-at 0) (labels-at 1))
  (check-equal? (crosstab (labels-at 0) species) (crosstab (labels-at 1) species))
  (check-equal? (labels-at 2) (labels-at 2))
  (check-equal? (crosstab (labels-at 2) species) '((50 0 0) (0 47 14) (0 3 36))))

(test-gpu "the guide: refusals name the procedure or the generic"
  (check-cuml-error 'logic
                    "kmeans-fit: X: 150 samples are fewer than the 151 clusters"
                    (lambda () (kmeans-fit (iris) #:n-clusters 151)))
  (check-cuml-error 'logic
                    "fit: X: 150 samples are fewer than the 151 clusters"
                    (lambda () (fit (kmeans #:n-clusters 151) (iris))))
  (check-exn exn:fail:contract? (lambda () (kmeans-fit (iris) #:n-clusters 3 #:n-init 2.5))))

(define wine-features (~> (load-dataset-table 'wine) dataset-table-names (remove "cultivar" _)))
(define cultivar (column-of 'wine "cultivar"))

(define (wine-raw columns)
  (matrix->device-matrix (matrix-of 'wine columns) #:dtype 'float32))

(test-gpu "the guide: k-means on raw wine follows proline, on scaled wine the cultivars"
  (check-equal? (sizes cultivar) '(71 59 48))
  (check-equal? (for/list ([name (in-list '("proline" "magnesium" "hue"))])
                  (define column (column-of 'wine name))
                  (list name (apply min column) (apply max column)))
                '(("proline" 278 1680) ("magnesium" 70 162) ("hue" 0.48 1.71)))
  (define raw-table (crosstab (labels-of (kmeans #:n-clusters 3) (wine-raw wine-features)) cultivar))
  (check-equal? raw-table '((46 1 0) (0 50 19) (13 20 29)))
  (check-equal? (agreement raw-table) 125)
  (check-equal? (crosstab (labels-of (kmeans #:n-clusters 3) (wine-raw '("proline"))) cultivar)
                raw-table)
  (define X (scaled 'wine wine-features))
  (define scaled-table (crosstab (labels-of (kmeans #:n-clusters 3) X) cultivar))
  (check-equal? scaled-table '((59 5 0) (0 63 0) (0 3 48)))
  (check-equal? (agreement scaled-table) 170)
  (check-equal? (for/list ([k (in-range 1 7)])
                  (define fitted (fit (kmeans #:n-clusters k) X))
                  (list k (exact-round (score fitted X))))
                '((1 -2314) (2 -1660) (3 -1280) (4 -1212) (5 -1102) (6 -1066))))

(define (eruption-summary labels)
  (list (sizes (filter exact-nonnegative-integer? labels)) (count negative? labels)))

(test-gpu "the guide: DBSCAN finds the two eruption groups and eight noise samples"
  (define X (scaled 'faithful '("eruptions" "waiting")))
  (check-equal? (shape X) '(272 2))
  (check-equal? (eruption-summary (labels-of (dbscan #:eps 0.3 #:min-samples 5) X)) '((168 96) 8))
  (check-equal? (for/list ([eps (in-list '(0.1 0.2 0.3 0.5))])
                  (cons eps (eruption-summary (labels-of (dbscan #:eps eps #:min-samples 5) X))))
                '((0.1 (71 42 7 6 6 6 4 4) 126) (0.2 (160 87) 25) (0.3 (168 96) 8) (0.5 (272) 0)))
  (check-equal? (~> (dbscan-fit X #:eps 0.3 #:min-samples 5)
                    dbscan-result-core-sample-indices
                    device-vector->list
                    length)
                252))

(test-gpu "the guide: single linkage finds setosa, then chains versicolor and virginica"
  (define (single-linkage-table k)
    (crosstab (labels-of (agglomerative-clustering #:n-clusters k) (iris)) species))
  (check-equal? (single-linkage-table 2) '((50 0 0) (0 50 50)))
  (check-equal? (single-linkage-table 3) '((50 0 0) (0 50 48) (0 0 2)))
  (check-equal? (agreement (single-linkage-table 3)) 102)
  (check-equal? (single-linkage-table 6) '((50 0 0) (0 46 46) (0 4 0) (0 0 1) (0 0 1) (0 0 2))))

(define tumour-features
  (~> (load-dataset-table 'breast-cancer) dataset-table-names (remove* '("diagnosis" "id") _)))
(define diagnosis (column-of 'breast-cancer "diagnosis"))

(test-gpu "the guide: spectral clustering beats k-means on the tumours"
  (check-equal? (sizes diagnosis) '(357 212))
  (define X (scaled 'breast-cancer tumour-features))
  (define (table-of estimator)
    (crosstab (labels-of estimator X) diagnosis))
  (define kmeans-table (table-of (kmeans #:n-clusters 2)))
  (define spectral-table (table-of (spectral-clustering #:n-clusters 2)))
  (check-equal? kmeans-table '((176 18) (36 339)))
  (check-equal? spectral-table '((182 6) (30 351)))
  (check-equal? (map agreement (list kmeans-table spectral-table)) '(515 533))
  (check-equal? (for/list ([k (in-list '(3 5 10 30))])
                  (list k
                        (agreement (table-of (spectral-clustering #:n-clusters 2 #:n-neighbors k)))))
                '((3 516) (5 536) (10 533) (30 535))))
