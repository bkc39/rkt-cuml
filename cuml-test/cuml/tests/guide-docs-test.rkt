#lang racket/base

(require (only-in cuml
                  adjusted-rand-score
                  agglomerative-clustering
                  current-batch-samples
                  dbscan
                  dbscan-fit
                  dbscan-result-core-sample-indices
                  estimator-params
                  estimator-set-params
                  fit
                  fit-predict
                  fit-transform
                  hdbscan
                  hdbscan-approximate-predict
                  hdbscan-fit
                  hdbscan-prediction-labels
                  kmeans
                  kmeans-fit
                  kmeans-model
                  kmeans-params
                  kmeans-result-centroids
                  kmeans-result-labels
                  kmeans-result?
                  kmeans-transform
                  pca
                  pca-fit
                  pca-result-components
                  pca-result-explained-variance
                  pca-result-explained-variance-ratio
                  pca-result-singular-values
                  predict
                  score
                  single-linkage-fit
                  spectral-clustering
                  table->device-matrix
                  tsvd-fit
                  tsvd-result-components
                  tsvd-result-explained-variance-ratio
                  tsvd-result-singular-values)
         (only-in cuml/scribblings/guide/helpers
                  class-ids
                  components-for
                  counts-of
                  cumulative
                  dataset-column
                  features
                  labels-of
                  leading
                  matched-labels
                  rounded
                  rows->matrix
                  sizes)
         (only-in cuml/scribblings/guide/source page-helpers-source)
         (only-in datasets/core load-dataset-table)
         (only-in pkg/lib pkg-directory)
         (only-in racket/file file->string)
         (only-in racket/function curry curryr)
         (only-in racket/list
                  append-map
                  argmax
                  count
                  drop
                  first
                  group-by
                  remove-duplicates
                  second
                  take)
         (only-in racket/match match match-define)
         (only-in racket/math exact-round)
         (only-in racket/string string-contains?)
         (only-in rackunit
                  check
                  check-=
                  check-equal?
                  check-exn
                  check-false
                  check-not-equal?
                  check-not-false
                  check-pred
                  test-case)
         (only-in raft
                  device-matrix->list*
                  device-resources
                  device-vector->list
                  dtype
                  list*->device-matrix
                  shape
                  with-device-resources)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in syntax/modread with-module-reading-parameterization)
         (only-in threading ~>)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/guide-data.rkt" cluster-columns host scaled-matrix table-ari)
         (only-in "private/scaler-data.rkt" unit-roundoff))

(define-namespace-anchor anchor)

(test-case "the page's require alone, in a racket/base module, expands the page's helpers and runs a threaded example"
  (define-values (shown-require shown-helpers) (page-helpers-source))
  (parameterize ([current-namespace (namespace-anchor->empty-namespace anchor)])
    (namespace-require 'racket/base)
    (eval `(module page racket/base
             ,(syntax->datum shown-require)
             ,@(map syntax->datum shown-helpers)))
    (dynamic-require ''page #f)
    (check-equal? (eval '(~> '(1 1 0) (matched-labels '(a a b))) (module->namespace ''page))
                  '(0 0 1))))

(define (defined-name form)
  (match (syntax->datum form)
    [(list* 'define (cons name _) _) name]
    [(list* 'define (? symbol? name) _) name]
    [_ #f]))

(test-case "the retired helpers are gone from the helpers module and from the page"
  (define-values (variables syntax) (module->exports 'cuml/scribblings/guide/helpers))
  (define exported (map car (append-map cdr (append variables syntax))))
  (define-values (_require shown-helpers) (page-helpers-source))
  (define defined (map defined-name shown-helpers))
  (check-not-false (memq 'matched-labels defined))
  (for ([name (in-list '(crosstab agreement
                                  standardized
                                  standardized-rows
                                  standardized-matrix
                                  dataset-matrix
                                  matrix-of))])
    (check-false (memq name exported) (symbol->string name))
    (check-false (memq name defined) (symbol->string name))))

(define (included-sections datum)
  (match datum
    [(list 'include-section (? string? section)) (list section)]
    [(cons head tail) (append (included-sections head) (included-sections tail))]
    [_ '()]))

(define (read-document file)
  (call-with-input-file file
                        (lambda (in) (with-module-reading-parameterization (lambda () (read in))))))

(define (guide-chapters [file (collection-file-path "guide.scrbl" "cuml" "scribblings")])
  (define-values (dir _name _directory?) (split-path file))
  (append-map (lambda (section)
                (define chapter (build-path dir section))
                (cons chapter (guide-chapters chapter)))
              (~> file read-document included-sections)))

(define dataset-citations
  '((iris ("doi:10.24432/C56C76" "CC BY 4.0") "doi:10.24432/C56C76. CC BY 4.0")
    (wine ("doi:10.24432/C5PC7J" "CC BY 4.0") "doi:10.24432/C5PC7J. CC BY 4.0")
    (faithful ("R 4.3.3" "GPL-2 or GPL-3") "R 4.3.3 datasets: GPL-2.0-only OR GPL-3.0-only")
    (breast-cancer ("doi:10.24432/C5DW2B" "CC BY 4.0") "doi:10.24432/C5DW2B. CC BY 4.0")))

(define (flattened-text path)
  (regexp-replace* #px"\\s+" (file->string path) " "))

(define dataset-use #px"\\((?:load-dataset-table|dataset-columns?|features) '([a-z-]+)")

(define (chapter-datasets chapter)
  (map string->symbol (regexp-match* dataset-use (flattened-text chapter) #:match-select cadr)))

(test-case "the guide chapter that introduces a dataset cites its DOI or licence, as datasets-core's NOTICE gives them"
  (define datasets-core (pkg-directory "datasets-core"))
  (check-not-false datasets-core "pkg-directory: datasets-core is not installed")
  (define notice (flattened-text (build-path datasets-core "NOTICE")))
  (define introductions
    (for*/fold ([introductions '()]
                #:result (reverse introductions))
               ([chapter (in-list (guide-chapters))]
                [dataset (in-list (remove-duplicates (chapter-datasets chapter)))]
                #:unless (assq dataset introductions))
      (cons (cons dataset chapter) introductions)))
  (check-equal? (map car introductions) (map car dataset-citations))
  (for ([citation (in-list dataset-citations)])
    (match-define (list dataset cited notice-text) citation)
    (define chapter (cdr (assq dataset introductions)))
    (check-not-false (string-contains? notice notice-text) (format "NOTICE: ~a" notice-text))
    (for ([text (in-list cited)])
      (check-not-false (string-contains? (flattened-text chapter) text)
                       (format "~a: ~a" chapter text)))))

(test-case "the helpers leave out noise, label -1"
  (check-equal? (sizes '(-1 0 0 1 -1)) '(2 1))
  (check-equal? (sizes '("B" "M" "B")) '(2 1)))

(define measurements '("sepal-length" "sepal-width" "petal-length" "petal-width"))

(define (flowers [element-type 'float32])
  (table->device-matrix (load-dataset-table 'iris) #:columns measurements #:dtype element-type))

(define species (dataset-column 'iris "species"))

(define (kmeans-labels result)
  (device-vector->list (kmeans-result-labels result)))

(define (check-scored ids labels columns text)
  (check-equal? (cluster-columns (counts-of ids labels)) columns)
  (define ari (adjusted-rand-score ids labels))
  (check-= ari (table-ari columns) 1e-12)
  (check-equal? (rounded (table-ari columns)) text)
  (check-equal? (rounded ari) text))

(define (species-ids)
  (class-ids species))

(define iris-kmeans-columns '((0 2 36) (0 48 14) (50 0 0)))

(test-gpu "the guide: iris on the device, and its three species"
  (check-equal? (features 'iris "species") measurements)
  (check-equal? (shape (flowers)) '(150 4))
  (check-equal? (remove-duplicates species) '("Iris-setosa" "Iris-versicolor" "Iris-virginica")))

(test-gpu "the guide: a first k-means against the species, and new flowers"
  (define fitted (fit (kmeans #:n-clusters 3) (flowers)))
  (check-equal? (format "~v" fitted) "#<kmeans fitted #:n-clusters 3>")
  (define labels (~> fitted kmeans-model kmeans-labels))
  (check-equal? (length labels) 150)
  (check-scored (species-ids)
                (~> fitted kmeans-model kmeans-result-labels)
                iris-kmeans-columns
                "0.73")
  (define cluster-species
    (for/hash ([label (in-list (remove-duplicates labels))])
      (define names
        (for/list ([sample-label (in-list labels)]
                   [name (in-list species)]
                   #:when (= sample-label label))
          name))
      (values label (~> names (group-by values _) (argmax length _) first))))
  (define new-flowers
    (list*->device-matrix '((5.0 3.4 1.5 0.2) (5.9 2.8 4.3 1.3) (6.6 3.0 5.6 2.0)) #:dtype 'float32))
  (check-equal? (map (curry hash-ref cluster-species)
                     (device-vector->list (predict fitted new-flowers)))
                '("Iris-setosa" "Iris-versicolor" "Iris-virginica")))

(define (labels-at seed)
  (kmeans-result-labels (kmeans-fit (flowers) #:n-clusters 3 #:random-state seed)))

(test-gpu "the guide: float64 for k-means, refused by single linkage, and col-major refused"
  (define flowers64 (flowers 'float64))
  (check-equal? (dtype flowers64) 'float64)
  (check-scored (species-ids)
                (kmeans-result-labels (kmeans-fit flowers64 #:n-clusters 3))
                iris-kmeans-columns
                "0.73")
  (check-cuml-error 'logic
                    "single-linkage-fit: X: expected float32, got float64"
                    (lambda () (single-linkage-fit flowers64 #:n-clusters 3)))
  (check-cuml-error
   'logic
   "kmeans-fit: X: expected row-major, got col-major 150x4"
   (lambda ()
     (~> (table->device-matrix (load-dataset-table 'iris) #:columns measurements #:layout 'col-major)
         (kmeans-fit #:n-clusters 3)))))

(test-gpu "the guide: scoped resources, the two forms, and renumbering seeds"
  (check-equal? (with-device-resources ([r (device-resources)])
                  (~> (kmeans-fit (flowers) #:n-clusters 3 #:resources r)
                      kmeans-result-labels
                      (adjusted-rand-score (species-ids) _ #:resources r)
                      rounded))
                "0.73")
  (define estimator (kmeans #:n-clusters 3))
  (check-equal? (device-vector->list (fit-predict estimator (flowers)))
                (device-vector->list (labels-at 0)))
  (check-equal? (hash-ref (estimator-params estimator) 'n-clusters) 3)
  (check-equal? (format "~v" (estimator-set-params estimator #:n-clusters 2))
                "#<kmeans #:n-clusters 2>")
  (check-pred kmeans-result? (kmeans-model (fit estimator (flowers))))
  (check-not-equal? (device-vector->list (labels-at 0)) (device-vector->list (labels-at 1)))
  (check-equal? (adjusted-rand-score (labels-at 0) (labels-at 1)) 1.0)
  (check-equal? (device-vector->list (labels-at 2)) (device-vector->list (labels-at 2)))
  (check-scored (species-ids) (labels-at 2) '((0 3 36) (0 47 14) (50 0 0)) "0.716"))

(test-gpu "the guide: a base set swept over k, changed by keyword, and the batch size"
  (define base (kmeans-params #:n-init 3))
  (for ([row (in-list '((2 ((0 47 50) (50 3 0)) "0.54") (3 ((0 2 36) (0 48 14) (50 0 0)) "0.73")
                                                        (4 ((0 0 32) (0 23 17) (0 27 1) (50 0 0))
                                                           "0.65")))])
    (match-define (list k columns text) row)
    (check-scored (species-ids)
                  (~> (fit (kmeans base #:n-clusters k) (flowers)) kmeans-model kmeans-result-labels)
                  columns
                  text))
  (check-equal? (hash-ref (estimator-params (estimator-set-params (kmeans base) #:n-clusters 2))
                          'n-init)
                3)
  (define (base-labels)
    (kmeans-labels (kmeans-fit (flowers) #:params base #:n-clusters 3)))
  (check-equal? (base-labels)
                (parameterize ([current-batch-samples 16])
                  (base-labels))))

(test-gpu "the guide: refusals name the procedure or the generic"
  (check-cuml-error 'logic
                    "kmeans-fit: X: 150 samples are fewer than the 151 clusters"
                    (lambda () (kmeans-fit (flowers) #:n-clusters 151)))
  (check-cuml-error 'logic
                    "fit: X: 150 samples are fewer than the 151 clusters"
                    (lambda () (fit (kmeans #:n-clusters 151) (flowers))))
  (check-exn exn:fail:contract? (lambda () (kmeans-fit (flowers) #:n-clusters 3 #:n-init 2.5))))

(define wine-measurements (features 'wine "cultivar"))
(define cultivar (dataset-column 'wine "cultivar"))

(define (cultivar-ids)
  (class-ids cultivar))

(define (cultivar-labels wines)
  (fit-predict (kmeans #:n-clusters 3) wines))

(define wine-scaled-columns '((0 3 48) (0 63 0) (59 5 0)))

(test-gpu "the guide: k-means on raw wine follows proline, on scaled wine the cultivars"
  (check-equal? (length wine-measurements) 13)
  (check-equal? (sizes cultivar) '(71 59 48))
  (check-equal? (for/list ([name (in-list '("proline" "magnesium" "hue"))])
                  (define column (dataset-column 'wine name))
                  (list name (apply min column) (apply max column)))
                '(("proline" 278 1680) ("magnesium" 70 162) ("hue" 0.48 1.71)))
  (define raw-labels
    (~> (load-dataset-table 'wine)
        (table->device-matrix #:columns wine-measurements)
        cultivar-labels))
  (check-scored (cultivar-ids) raw-labels '((0 50 19) (13 20 29) (46 1 0)) "0.371")
  (check-equal? (~> (load-dataset-table 'wine)
                    (table->device-matrix #:columns '("proline"))
                    cultivar-labels
                    (adjusted-rand-score raw-labels _))
                1.0)
  (define scaled (scaled-matrix 'wine wine-measurements))
  (check-scored (cultivar-ids) (cultivar-labels scaled) wine-scaled-columns "0.864")
  (define scores
    (for/list ([k (in-range 1 7)])
      (exact-round (score (fit (kmeans #:n-clusters k #:n-init 10) scaled) scaled))))
  (check-equal? scores '(-2314 -1659 -1278 -1176 -1102 -1053)))

(define (eruptions)
  (scaled-matrix 'faithful '("eruptions" "waiting")))

(test-gpu "the guide: HDBSCAN finds the two eruption groups with no eps to choose"
  (define labels (labels-of (hdbscan) (eruptions)))
  (check-equal? (sizes labels) '(174 94))
  (check-equal? (count negative? labels) 4)
  (define scaled (device-matrix->list* (eruptions)))
  (define predicted
    (~> (hdbscan-fit (rows->matrix (take scaled 200)))
        (hdbscan-approximate-predict (rows->matrix (drop scaled 200)))
        hdbscan-prediction-labels
        device-vector->list))
  (check-equal? (count negative? predicted) 2)
  (check-equal? (sizes predicted) '(45 25)))

(define (eruption-clusters-and-noise eps)
  (define labels (labels-of (dbscan #:eps eps #:min-samples 5) (eruptions)))
  (list (length (sizes labels)) (count negative? labels)))

(test-gpu "the guide: DBSCAN finds the two eruption groups and eight noise samples"
  (check-equal? (shape (eruptions)) '(272 2))
  (check-equal? (for/list ([eps (in-list '(0.1 0.2 0.3 0.5))])
                  (cons eps (eruption-clusters-and-noise eps)))
                '((0.1 8 126) (0.2 2 25) (0.3 2 8) (0.5 1 0)))
  (check-equal? (~> (dbscan-fit (eruptions) #:eps 0.3 #:min-samples 5)
                    dbscan-result-core-sample-indices
                    device-vector->list
                    length)
                252))

(test-gpu "the guide: single linkage finds setosa, then chains versicolor and virginica"
  (define (single-linkage-labels k)
    (fit-predict (agglomerative-clustering #:n-clusters k) (flowers)))
  (check-equal? (cluster-columns (counts-of (species-ids) (single-linkage-labels 2)))
                '((0 50 50) (50 0 0)))
  (check-scored (species-ids) (single-linkage-labels 3) '((0 0 2) (0 50 48) (50 0 0)) "0.564")
  (define six (cluster-columns (counts-of (species-ids) (single-linkage-labels 6))))
  (check-equal? (length six) 6)
  (check-not-false (member '(0 46 46) six)))

(define tumour-measurements (features 'breast-cancer "diagnosis" "id"))
(define diagnosis (dataset-column 'breast-cancer "diagnosis"))

(test-gpu "the guide: spectral clustering beats k-means on the tumours"
  (check-equal? (length tumour-measurements) 30)
  (check-equal? (sizes diagnosis) '(357 212))
  (define tumours (scaled-matrix 'breast-cancer tumour-measurements))
  (define diagnosis-ids (class-ids diagnosis))
  (define (labels-in estimator)
    (fit-predict estimator tumours))
  (check-scored diagnosis-ids (labels-in (kmeans #:n-clusters 2)) '((36 339) (176 18)) "0.654")
  (check-scored diagnosis-ids
                (labels-in (spectral-clustering #:n-clusters 2))
                '((30 351) (182 6))
                "0.761")
  (for ([row (in-list '((3 ((43 347) (169 10)) "0.659") (5 ((32 356) (180 1)) "0.779")
                                                        (10 ((30 351) (182 6)) "0.761")
                                                        (30 ((27 350) (185 7)) "0.773")))])
    (match-define (list k columns text) row)
    (check-scored diagnosis-ids
                  (labels-in (spectral-clustering #:n-clusters 2 #:n-neighbors k))
                  columns
                  text)))

(define (wine-scaled)
  (scaled-matrix 'wine wine-measurements))

(test-gpu "the guide: decomposition, raw wine is proline and scaled wine needs five components for 80%"
  (define raw-fit
    (~> (load-dataset-table 'wine)
        (table->device-matrix #:columns wine-measurements)
        (pca-fit #:n-components 2)))
  (check-equal? (~> raw-fit pca-result-explained-variance-ratio device-vector->list (map rounded _))
                '("0.998" "0.002"))
  (check-equal? (leading raw-fit wine-measurements 1) '(("proline") ("magnesium")))
  (define scaled-fit (pca-fit (wine-scaled)))
  (define ratios (~> scaled-fit pca-result-explained-variance-ratio device-vector->list))
  (check-equal? (map rounded (take ratios 3)) '("0.362" "0.192" "0.111"))
  (check-equal? (~> ratios cumulative second rounded) "0.554")
  (check-equal? (map (curry components-for ratios) '(0.8 0.9)) '(5 8))
  (check-equal? (take (leading scaled-fit wine-measurements 2) 2)
                '(("flavanoids" "total-phenols") ("color-intensity" "alcohol"))))

(define (clusterer [seed 0])
  (kmeans #:n-clusters 3 #:n-init 10 #:random-state seed))

(define (centroid-gaps fitted samples)
  (~> (kmeans-transform (~> fitted kmeans-model kmeans-result-centroids) samples)
      device-matrix->list*
      (map (lambda (distances)
             (match (sort (map sqrt distances) <)
               [(list* nearest next _) (- next nearest)]))
           _)))

(define (projections)
  (define scaled (wine-scaled))
  (list scaled
        (fit-transform (pca #:n-components 2) scaled)
        (fit-transform (pca #:n-components 2 #:whiten #t) scaled)))

(test-gpu "the guide: decomposition, two components cluster about as well as 13 columns, whitened better"
  (define samples (projections))
  (match-define (list all-labels projected-labels whitened-labels)
    (map (curry fit-predict (clusterer)) samples))
  (check-scored (cultivar-ids) all-labels '((0 3 48) (0 65 0) (59 3 0)) "0.897")
  (check-scored (cultivar-ids) projected-labels '((0 1 48) (0 65 0) (59 5 0)) "0.895")
  (check-scored (cultivar-ids) whitened-labels '((0 1 48) (1 67 0) (58 3 0)) "0.912")
  (for ([labels (in-list (list projected-labels whitened-labels))]
        [reduced (in-list (cdr samples))])
    (check >= (adjusted-rand-score all-labels labels) 0.85)
    (define fitted (fit (clusterer) reduced))
    (check-equal? (~> fitted kmeans-model kmeans-result-labels device-vector->list)
                  (device-vector->list labels))
    (check >= (apply min (centroid-gaps fitted reduced)) 1e-3)))

(test-gpu "the guide: decomposition, the k-means ordering holds for seeds 0 to 9"
  (define samples (projections))
  (for ([seed (in-range 10)])
    (match-define (list all projected whitened)
      (map (lambda (reduced)
             (adjusted-rand-score (cultivar-ids) (fit-predict (clusterer seed) reduced)))
           samples))
    (check >= whitened (max all projected) (format "seed ~a" seed))
    (check <= (abs (- projected all)) 0.015 (format "seed ~a" seed))))

(define u (unit-roundoff 'float32))

(define (worst-difference xs ys)
  (apply max (map (lambda (x y) (abs (- x y))) xs ys)))

(test-gpu "the guide: decomposition, truncated SVD of the centred wines is PCA's"
  (define scaled (wine-scaled))
  (define svd-fit (tsvd-fit scaled #:n-components 2))
  (define scaled-fit (pca-fit scaled))
  (define (shown result singular-values ratios)
    (~> (list (singular-values result) (ratios result))
        (map (compose (curry map rounded) (curryr host 2)) _)))
  (check-equal? (shown svd-fit tsvd-result-singular-values tsvd-result-explained-variance-ratio)
                '(("28.942" "21.082") ("0.362" "0.192")))
  (check-equal? (shown scaled-fit pca-result-singular-values pca-result-explained-variance-ratio)
                '(("28.942" "21.082") ("0.362" "0.192")))
  (match-define (list largest second-largest third-largest)
    (host (pca-result-explained-variance scaled-fit) 3))
  (match-define (list svd-singular pca-singular)
    (map (curryr host 2)
         (list (tsvd-result-singular-values svd-fit) (pca-result-singular-values scaled-fit))))
  (check <= (worst-difference svd-singular pca-singular) (* 64 u (first pca-singular)))
  (check <=
         (worst-difference (host (tsvd-result-explained-variance-ratio svd-fit) 2)
                           (host (pca-result-explained-variance-ratio scaled-fit) 2))
         (* 64 u))
  (for ([svd-component (in-list (device-matrix->list* (tsvd-result-components svd-fit)))]
        [pca-component (in-list (device-matrix->list* (pca-result-components scaled-fit)))]
        [gap (in-list (list (- largest second-largest)
                            (min (- largest second-largest) (- second-largest third-largest))))])
    (check <= (worst-difference svd-component pca-component) (/ (* 64 u largest) gap))))
