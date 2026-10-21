#lang racket/base

(require (only-in cuml
                  agglomerative-clustering
                  dbscan-fit
                  dbscan-result-core-sample-indices
                  dbscan-result-labels
                  fit
                  kmeans
                  kmeans-model
                  kmeans-result-labels
                  spectral-clustering)
         (only-in cuml/scribblings/guide/helpers
                  cluster-colors
                  cluster-legend
                  cluster-plot
                  cluster-style
                  dataset-column
                  dataset-matrix
                  features
                  labels-of
                  line-plot
                  matched-labels
                  panels
                  renumbered
                  sizes
                  standardized-matrix)
         (only-in pict pict->bitmap pict-height pict-width)
         (only-in racket/class send)
         (only-in racket/list count range remove-duplicates)
         (only-in rackunit check-equal? check-exn check-false check-not-false test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>))

(define (pixels picture)
  (define bitmap (pict->bitmap picture))
  (define width (send bitmap get-width))
  (define height (send bitmap get-height))
  (define argb (make-bytes (* 4 width height)))
  (send bitmap get-argb-pixels 0 0 width height argb)
  (for/list ([offset (in-range 0 (bytes-length argb) 4)])
    (list (bytes-ref argb (+ offset 1)) (bytes-ref argb (+ offset 2)) (bytes-ref argb (+ offset 3)))))

(define (blended color)
  (map (lambda (channel) (+ (* 4/5 channel) (* 1/5 255))) color))

(define (drawn? picture target)
  (for/or ([pixel (in-list (pixels picture))])
    (for/and ([channel (in-list pixel)]
              [expected (in-list target)])
      (<= (abs (- channel expected)) 2))))

(define one-sample (cluster-plot '(0.0) '(0.0) '(1)))

(test-case "cluster-style gives each label id its colour and marker, in the fixed order"
  (check-equal? (map cluster-style (range 8))
                '(((42 120 214) fullcircle circle) ((235 104 52) fullsquare square)
                                                   ((27 175 122) fulltriangleup triangleup)
                                                   ((237 161 0) fulldiamond diamond)
                                                   ((232 123 164) fulltriangledown triangledown)
                                                   ((0 131 0) full5star 5star)
                                                   ((98 80 214) fulltriangleright triangleright)
                                                   ((227 73 72) full4star 4star)))
  (check-equal? (length (remove-duplicates (map cadr (map cluster-style (range 8))))) 8)
  (check-exn #rx"^cluster-plot: a figure has at most 8 clusters\n  label: 8"
             (lambda () (cluster-plot '(0 1) '(0 1) '(0 8)))))

(test-case "a label keeps its colour whatever other labels a panel holds"
  (check-not-false (drawn? one-sample (blended (cadr cluster-colors))))
  (check-false (drawn? one-sample (blended (car cluster-colors))))
  (check-not-false (drawn? (cluster-plot '(0.0 1.0) '(0.0 1.0) '(0 1))
                           (blended (cadr cluster-colors)))))

(test-case "the plots are picts of the requested size, panels lay them out in rows"
  (check-equal? (list (pict-width one-sample) (pict-height one-sample)) '(320 280))
  (define small (cluster-plot '(0 1) '(0 1) '(0 0) #:width 200 #:height 150))
  (check-equal? (list (pict-width small) (pict-height small)) '(200 150))
  (define line (line-plot '(1 2 3) '(-3 -2 -1)))
  (check-equal? (list (pict-width line) (pict-height line)) '(320 240))
  (define figure (panels (list one-sample one-sample) (list one-sample)))
  (check-equal? (list (pict-width figure) (pict-height figure)) '(652 572))
  (check-equal? (map pict-height
                     (list (cluster-legend '(0 1 2))
                           (cluster-legend '(0 1 2 3) #:centroids? #t)
                           (cluster-legend '(-1 0 1 2 3 4 5 6 7))))
                '(38 60 82))
  (check-equal? (pict-width (cluster-legend '(0) #:width 400)) 400))

(test-case "all noise, one cluster, and mismatched lengths"
  (define noise (cluster-plot '(1 2 3) '(1 2 3) '(-1 -1 -1)))
  (check-equal? (list (pict-width noise) (pict-height noise)) '(320 280))
  (check-not-false (drawn? noise '(137 135 129)))
  (check-false (drawn? noise (blended (car cluster-colors))))
  (check-not-false (drawn? (cluster-plot '(1 2 3) '(1 2 3) '(0 0 0)) (blended (car cluster-colors))))
  (check-exn #rx"^cluster-plot: xs, ys and labels differ in length\n  lengths: '[(]3 2[)]"
             (lambda () (cluster-plot '(1 2 3) '(1 2 3) '(0 0)))))

(test-case "renumbered numbers values by first appearance and keeps noise"
  (check-equal? (renumbered '("b" "a" "b" "c")) '(0 1 0 2))
  (check-equal? (renumbered '(3 -1 3 1)) '(0 -1 0 1)))

(test-case "matched-labels gives a cluster its class's number, largest overlap first, the rest after"
  (check-equal? (matched-labels '(2 2 0 0 1 1 -1) '(a a b b c c a)) '(0 0 1 1 2 2 -1))
  (check-equal? (matched-labels '(5 5 7 7 7 9) '(x x y y y y)) '(0 0 1 1 1 2))
  (check-equal? (matched-labels '(4 4 4 4 6 6) '(x x y y y y)) '(0 0 0 0 1 1))
  (check-equal? (matched-labels '(0 0 1) '(x x x)) '(0 0 1))
  (check-equal? (matched-labels '(3 3 4) '(-1 -1 7)) '(1 1 0)))

(define (mismatches labels classes)
  (count (lambda (cluster class) (not (= cluster class))) labels (renumbered classes)))

(define species (dataset-column 'iris "species"))

(test-gpu "the figures: k-means on iris differs from the species on the 16 flowers the prose names"
  (define labels
    (~> (fit (kmeans #:n-clusters 3) (dataset-matrix 'iris (features 'iris "species")))
        kmeans-model
        kmeans-result-labels
        device-vector->list))
  (check-equal? (mismatches (matched-labels labels species) species) 16))

(test-gpu "the figures: wine clusters match the cultivars but for 53 raw and 8 standardized wines"
  (define cultivar (dataset-column 'wine "cultivar"))
  (define measurements (features 'wine "cultivar"))
  (define (clusters-of wines)
    (matched-labels (labels-of (kmeans #:n-clusters 3) wines) cultivar))
  (check-equal? (remove-duplicates cultivar) '("1" "2" "3"))
  (check-equal? (mismatches (clusters-of (dataset-matrix 'wine measurements)) cultivar) 53)
  (check-equal? (mismatches (clusters-of (standardized-matrix 'wine measurements)) cultivar) 8))

(test-gpu "the figures: DBSCAN's groups keep their numbers across the radii"
  (define eruptions (standardized-matrix 'faithful '("eruptions" "waiting")))
  (define fits
    (for/list ([eps (in-list '(0.1 0.2 0.3 0.5))])
      (dbscan-fit eruptions #:eps eps #:min-samples 5)))
  (define (labels-in result)
    (device-vector->list (dbscan-result-labels result)))
  (define groups (labels-in (caddr fits)))
  (define matched
    (for/list ([result (in-list fits)])
      (matched-labels (labels-in result) groups)))
  (check-equal? (map (lambda (labels) (sort (remove-duplicates labels) <)) matched)
                '((-1 0 1 2 3 4 5 6 7) (-1 0 1) (-1 0 1) (0)))
  (check-equal? (map (lambda (labels) (count negative? labels)) matched) '(126 25 8 0))
  (check-equal? (for/list ([labels (in-list matched)])
                  (for/list ([group (in-list '(0 1))])
                    (count (lambda (label reference) (and (= label group) (= reference group)))
                           labels
                           (renumbered groups))))
                '((71 42) (160 87) (168 96) (168 0))))

(test-gpu "the figures: single linkage keeps versicolor and virginica in one orange cluster"
  (define flowers (dataset-matrix 'iris (features 'iris "species")))
  (define cuts
    (for/list ([k (in-list '(2 3 4 6))])
      (matched-labels (labels-of (agglomerative-clustering #:n-clusters k) flowers) species)))
  (check-equal? (map sizes cuts) '((100 50) (98 50 2) (97 50 2 1) (92 50 4 2 1 1)))
  (check-equal? (map (lambda (cut) (count (lambda (label) (= label 1)) cut)) cuts) '(100 98 97 92)))

(test-gpu "the figures: the tumour clusters differ from the diagnosis on 54 and 36 tumours"
  (define diagnosis (dataset-column 'breast-cancer "diagnosis"))
  (define tumours (standardized-matrix 'breast-cancer (features 'breast-cancer "diagnosis" "id")))
  (define (clusters-of estimator)
    (matched-labels (labels-of estimator tumours) diagnosis))
  (check-equal? (remove-duplicates diagnosis) '("M" "B"))
  (check-equal? (mismatches (clusters-of (kmeans #:n-clusters 2)) diagnosis) 54)
  (check-equal? (mismatches (clusters-of (spectral-clustering #:n-clusters 2)) diagnosis) 36))
