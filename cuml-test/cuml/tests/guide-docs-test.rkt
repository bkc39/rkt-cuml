#lang racket/base

(require (only-in cuml
                  agglomerative-clustering
                  current-batch-samples
                  dbscan
                  dbscan-fit
                  dbscan-result-core-sample-indices
                  estimator-params
                  estimator-set-params
                  fit
                  fit-predict
                  hdbscan
                  hdbscan-approximate-predict
                  hdbscan-fit
                  hdbscan-prediction-labels
                  kmeans
                  kmeans-fit
                  kmeans-model
                  kmeans-params
                  kmeans-result-labels
                  kmeans-result?
                  predict
                  score
                  single-linkage-fit
                  spectral-clustering
                  standard-scaler-fit-transform
                  table->device-matrix)
         (only-in cuml/scribblings/guide/helpers
                  agreement
                  crosstab
                  dataset-column
                  features
                  labels-of
                  rows->matrix
                  sizes)
         (only-in cuml/scribblings/guide/source page-helpers-source)
         (only-in datasets/core load-dataset-table)
         (only-in pkg/lib pkg-directory)
         (only-in racket/file file->string)
         (only-in racket/function curry)
         (only-in racket/list append-map argmax count drop first group-by remove-duplicates take)
         (only-in racket/match match match-define)
         (only-in racket/math exact-round)
         (only-in racket/string string-contains?)
         (only-in rackunit
                  check-equal?
                  check-exn
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
         (only-in "private/cuml-error.rkt" check-cuml-error))

(define-namespace-anchor anchor)

(define (scaled-matrix name columns)
  (define-values (_fitted scaled)
    (~> (load-dataset-table name) (table->device-matrix #:columns columns) standard-scaler-fit-transform))
  scaled)

(test-case "the page's require alone, in a racket/base module, expands the page's helpers and runs a threaded example"
  (define-values (shown-require shown-helpers) (page-helpers-source))
  (parameterize ([current-namespace (namespace-anchor->empty-namespace anchor)])
    (namespace-require 'racket/base)
    (eval `(module page racket/base
             ,(syntax->datum shown-require)
             ,@(map syntax->datum shown-helpers)))
    (dynamic-require ''page #f)
    (check-equal? (eval '(~> '(0 0 1) (crosstab '(a a b)) agreement) (module->namespace ''page)) 3)))

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

(define dataset-use
  #px"\\((?:load-dataset-table|dataset-columns?|features) '([a-z-]+)")

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
  (define labels '(0 0 -1 1 1 -1))
  (define classes '(a a a b b b))
  (check-equal? (crosstab labels classes) '((2 0) (0 2)))
  (check-equal? (agreement (crosstab labels classes)) 4)
  (check-equal? (sizes '(-1 0 0 1 -1)) '(2 1))
  (check-equal? (sizes '("B" "M" "B")) '(2 1)))

(define measurements '("sepal-length" "sepal-width" "petal-length" "petal-width"))

(define (flowers [element-type 'float32])
  (table->device-matrix (load-dataset-table 'iris) #:columns measurements #:dtype element-type))

(define species (dataset-column 'iris "species"))

(define (kmeans-labels result)
  (device-vector->list (kmeans-result-labels result)))

(define iris-kmeans-table '((50 0 0) (0 48 14) (0 2 36)))

(test-gpu "the guide: iris on the device, and its three species"
  (check-equal? (features 'iris "species") measurements)
  (check-equal? (shape (flowers)) '(150 4))
  (check-equal? (remove-duplicates species) '("Iris-setosa" "Iris-versicolor" "Iris-virginica")))

(test-gpu "the guide: a first k-means against the species, and new flowers"
  (define fitted (fit (kmeans #:n-clusters 3) (flowers)))
  (check-equal? (format "~v" fitted) "#<kmeans fitted #:n-clusters 3>")
  (define labels (~> fitted kmeans-model kmeans-labels))
  (check-equal? (length labels) 150)
  (check-equal? (crosstab labels species) iris-kmeans-table)
  (check-equal? (agreement (crosstab labels species)) 134)
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
  (kmeans-labels (kmeans-fit (flowers) #:n-clusters 3 #:random-state seed)))

(test-gpu "the guide: float64 for k-means, refused by single linkage, and col-major refused"
  (define flowers64 (flowers 'float64))
  (check-equal? (dtype flowers64) 'float64)
  (check-equal? (~> (kmeans-fit flowers64 #:n-clusters 3) kmeans-labels (crosstab species))
                iris-kmeans-table)
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
                      kmeans-labels
                      (crosstab species)
                      agreement))
                134)
  (define estimator (kmeans #:n-clusters 3))
  (check-equal? (device-vector->list (fit-predict estimator (flowers))) (labels-at 0))
  (check-equal? (hash-ref (estimator-params estimator) 'n-clusters) 3)
  (check-equal? (format "~v" (estimator-set-params estimator #:n-clusters 2))
                "#<kmeans #:n-clusters 2>")
  (check-pred kmeans-result? (kmeans-model (fit estimator (flowers))))
  (check-not-equal? (labels-at 0) (labels-at 1))
  (check-equal? (crosstab (labels-at 0) species) (crosstab (labels-at 1) species))
  (check-equal? (labels-at 2) (labels-at 2))
  (check-equal? (crosstab (labels-at 2) species) '((50 0 0) (0 47 14) (0 3 36))))

(test-gpu "the guide: a base set swept over k, changed by keyword, and the batch size"
  (define base (kmeans-params #:n-init 3))
  (check-equal? (for/list ([k (in-range 2 5)])
                  (~> (fit (kmeans base #:n-clusters k) (flowers))
                      kmeans-model
                      kmeans-labels
                      (crosstab species)
                      agreement))
                '(100 134 132))
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

(define (cultivar-table wines)
  (crosstab (labels-of (kmeans #:n-clusters 3) wines) cultivar))

(test-gpu "the guide: k-means on raw wine follows proline, on scaled wine the cultivars"
  (check-equal? (length wine-measurements) 13)
  (check-equal? (sizes cultivar) '(71 59 48))
  (check-equal? (for/list ([name (in-list '("proline" "magnesium" "hue"))])
                  (define column (dataset-column 'wine name))
                  (list name (apply min column) (apply max column)))
                '(("proline" 278 1680) ("magnesium" 70 162) ("hue" 0.48 1.71)))
  (define raw-table
    (~> (load-dataset-table 'wine) (table->device-matrix #:columns wine-measurements) cultivar-table))
  (check-equal? raw-table '((46 1 0) (0 50 19) (13 20 29)))
  (check-equal? (agreement raw-table) 125)
  (check-equal?
   (~> (load-dataset-table 'wine) (table->device-matrix #:columns '("proline")) cultivar-table)
   raw-table)
  (define scaled (scaled-matrix 'wine wine-measurements))
  (define scaled-table (cultivar-table scaled))
  (check-equal? scaled-table '((59 5 0) (0 63 0) (0 3 48)))
  (check-equal? (agreement scaled-table) 170)
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
  (define (single-linkage-table k)
    (crosstab (labels-of (agglomerative-clustering #:n-clusters k) (flowers)) species))
  (check-equal? (single-linkage-table 2) '((50 0 0) (0 50 50)))
  (check-equal? (single-linkage-table 3) '((50 0 0) (0 50 48) (0 0 2)))
  (check-equal? (agreement (single-linkage-table 3)) 102)
  (define six (single-linkage-table 6))
  (check-equal? (length six) 6)
  (check-not-false (member '(0 46 46) six)))

(define tumour-measurements (features 'breast-cancer "diagnosis" "id"))
(define diagnosis (dataset-column 'breast-cancer "diagnosis"))

(test-gpu "the guide: spectral clustering beats k-means on the tumours"
  (check-equal? (length tumour-measurements) 30)
  (check-equal? (sizes diagnosis) '(357 212))
  (define tumours (scaled-matrix 'breast-cancer tumour-measurements))
  (define (table-of estimator)
    (crosstab (labels-of estimator tumours) diagnosis))
  (define kmeans-table (table-of (kmeans #:n-clusters 2)))
  (define spectral-table (table-of (spectral-clustering #:n-clusters 2)))
  (check-equal? kmeans-table '((176 18) (36 339)))
  (check-equal? spectral-table '((182 6) (30 351)))
  (check-equal? (map agreement (list kmeans-table spectral-table)) '(515 533))
  (check-equal? (for/list ([k (in-list '(3 5 10 30))])
                  (~> (spectral-clustering #:n-clusters 2 #:n-neighbors k) table-of agreement))
                '(516 536 533 535)))
