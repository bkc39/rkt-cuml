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
                  spectral-clustering
                  table->device-matrix)
         (only-in cuml/scribblings/guide/helpers
                  class-figure
                  cluster-colors
                  cluster-legend
                  cluster-plot
                  cluster-style
                  clusters-of
                  dataset-column
                  features
                  line-plot
                  matched-labels
                  renumbered
                  sizes
                  standardized-matrix
                  sweep-figure)
         (only-in datasets/core load-dataset-table)
         (only-in pict pict->bitmap pict-height pict-width pict?)
         (only-in racket/class send)
         (only-in racket/function curry)
         (only-in racket/list count first make-list range remove-duplicates rest second third)
         (only-in racket/match match-define)
         (only-in rackunit
                  check
                  check-equal?
                  check-exn
                  check-false
                  check-not-equal?
                  check-not-false
                  check-true
                  test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>))

(match-define (list* blue orange _) cluster-colors)

(define ink '(11 11 11))

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

(define (painted picture target)
  (count (lambda (pixel)
           (for/and ([channel (in-list pixel)]
                     [expected (in-list target)])
             (<= (abs (- channel expected)) 2)))
         (pixels picture)))

(define (drawn? picture target)
  (positive? (painted picture target)))

(define (size picture)
  (list (pict-width picture) (pict-height picture)))

(define one-sample (cluster-plot '(0.0) '(0.0) '(1)))

(test-case "cluster-style gives each label id its colour and marker, in the fixed order"
  (check-equal? (map cluster-style (range 8))
                (list '((42 120 214) fullcircle circle)
                      '((235 104 52) fullsquare square)
                      '((27 175 122) fulltriangleup triangleup)
                      '((237 161 0) fulldiamond diamond)
                      '((232 123 164) fulltriangledown triangledown)
                      '((0 131 0) full5star 5star)
                      '((98 80 214) fulltriangleright triangleright)
                      '((227 73 72) full4star 4star)))
  (check-equal? (~> (range 8) (map cluster-style _) (map second _) remove-duplicates length) 8)
  (check-exn #rx"^cluster-plot: label ids run from 0 to 7\n  label: 8"
             (lambda () (cluster-plot '(0 1) '(0 1) '(0 8)))))

(test-case "a negative label other than -1, or a non-integer one, is refused in the caller's name"
  (check-exn #rx"^cluster-plot: label ids run from 0 to 7\n  label: -2"
             (lambda () (cluster-plot '(0) '(0) '(-2))))
  (check-exn #rx"^cluster-legend: label ids run from 0 to 7\n  label: -2"
             (lambda () (cluster-legend '(-2))))
  (check-exn #rx"^cluster-legend: label ids run from 0 to 7\n  label: 1.0"
             (lambda () (cluster-legend '(1.0) #:names '("a" "b"))))
  (check-exn #rx"^cluster-style: label ids run from 0 to 7\n  label: 1.0"
             (lambda () (cluster-style 1.0))))

(test-case "a label keeps its colour whatever other labels a panel holds"
  (check-not-false (drawn? one-sample (blended orange)))
  (check-false (drawn? one-sample (blended blue)))
  (check-not-false (drawn? (cluster-plot '(0.0 1.0) '(0.0 1.0) '(0 1)) (blended orange))))

(test-case "core samples are filled, the others hollow; centroids are crosses"
  (define (blue-pixels core)
    (painted (cluster-plot '(0.0) '(0.0) '(0) #:core core) (blended blue)))
  (check > (blue-pixels '(0)) (* 2 (blue-pixels '())))
  (check-equal? (blue-pixels '(0)) (painted (cluster-plot '(0.0) '(0.0) '(0)) (blended blue)))
  (check-false (drawn? (cluster-plot '(0.0 1.0) '(0.0 1.0) '(0 0)) ink))
  (check-true (drawn? (cluster-plot '(0.0 1.0) '(0.0 1.0) '(0 0) #:centroids '((0.5 0.5))) ink)))

(test-case "the plots are picts of the requested size, the figures lay them out in rows of two"
  (check-equal? (size one-sample) '(320 280))
  (check-equal? (size (cluster-plot '(0 1) '(0 1) '(0 0) #:width 200 #:height 150)) '(200 150))
  (check-equal? (size (line-plot '(1 2 3) '(-3 -2 -1))) '(320 240))
  (check-equal? (size (class-figure '(0.0) '(0.0) (list (list "a" '(0))) '("x") #:class-title "x"))
                '(652 330))
  (check-equal?
   (size (class-figure '(0.0) '(0.0) (list (list "a" '(0)) (list "b" '(0))) '("x") #:class-title "x"))
   '(652 622))
  (check-equal? (size (sweep-figure '(0.0) '(0.0) '("a" "b" "c" "d") (make-list 4 '(0)))) '(652 582))
  (check-equal? (map pict-height
                     (list (cluster-legend '(0 1 2))
                           (cluster-legend '(0 1 2 3) #:centroids? #t)
                           (cluster-legend '(-1 0 1 2 3 4 5 6 7))))
                '(38 60 82))
  (check-equal? (pict-width (cluster-legend '(0) #:width 400)) 400)
  (check-equal? (size (cluster-legend '(0 1 2) #:width 100)) '(100 82))
  (check-equal? (size (cluster-legend '(0 1 2 3) #:names '("a" "b"))) '(652 38)))

(define four-xs '(0.0 1.0 2.0 3.0))

(define four-classes '("p" "q" "r" "s"))

(define (four-figure clusterings)
  (class-figure four-xs four-xs clusterings four-classes #:class-title "x"))

(test-case "class-figure names numeric classes"
  (check-true
   (pict?
    (class-figure '(0.0 1.0) '(0.0 1.0) (list (list "a" '(0 1))) '(0.0 1.0) #:class-title "x"))))

(test-case "class-figure's legend keys every label a panel draws, leftover clusters and noise"
  (define matched (pict-height (four-figure (list (list "a" '(0 1 2 3))))))
  (check-equal? (pict-height (four-figure (list (list "a" '(0 1 2 4))))) (+ matched 22))
  (check-equal? (pict-height (four-figure (list (list "a" '(0 1 2 -1))))) (+ matched 22)))

(test-case "class-figure's default names skip noise among the classes"
  (define (noisy draw)
    (pixels (draw '(0.0 1.0 2.0) '(0.0 1.0 2.0) (list (list "a" '(0 1 1))) '(-1 5 7))))
  (define ((named names) . arguments)
    (apply class-figure arguments #:class-title "x" #:names names))
  (define default (noisy (lambda arguments (apply class-figure arguments #:class-title "x"))))
  (check-equal? default (noisy (named '("5" "7"))))
  (check-not-equal? default (noisy (named '("-1" "5")))))

(test-case "class-figure draws a clustering's centroids and keys them"
  (define plain (four-figure (list (list "a" '(0 1 2 3)))))
  (define crossed (four-figure (list (list "a" '(0 1 2 3) '((1.5 1.5))))))
  (check-false (drawn? plain ink))
  (check > (painted crossed ink) (painted (cluster-legend '(0 1 2 3) #:centroids? #t) ink))
  (check-equal? (pict-height crossed) (+ (pict-height plain) 22)))

(test-case "sweep-figure refuses titles, clusterings and cores of different lengths"
  (check-exn #rx"^sweep-figure: titles, clusterings and cores differ in length\n  lengths: '[(]2 1[)]"
             (lambda () (sweep-figure '(0.0) '(0.0) '("a" "b") (list '(0)))))
  (check-exn #rx"^sweep-figure: titles, clusterings and cores differ in length\n  lengths: '[(]1 2[)]"
             (lambda () (sweep-figure '(0.0) '(0.0) '("a") (list '(0)) #:cores '(#f #f)))))

(test-case "all noise, one cluster, and mismatched lengths"
  (define noise (cluster-plot '(1 2 3) '(1 2 3) '(-1 -1 -1)))
  (check-equal? (size noise) '(320 280))
  (check-not-false (drawn? noise '(137 135 129)))
  (check-false (drawn? noise (blended blue)))
  (check-not-false (drawn? (cluster-plot '(1 2 3) '(1 2 3) '(0 0 0)) (blended blue)))
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

(test-case "seven clusters against three classes reach label id 8, past the figure's 0 to 7"
  (define labels (matched-labels '(0 0 0 0 1 2 3 4 5 6) '(a a b c a a a a a a)))
  (check-equal? labels '(0 0 0 0 3 4 5 6 7 8))
  (check-exn #rx"^cluster-plot: label ids run from 0 to 7\n  label: 8"
             (lambda () (cluster-plot (range 10) (range 10) labels))))

(define past-seven-classes '(a a b c a a a a a a))

(define past-seven (matched-labels '(0 0 0 0 1 2 3 4 5 6) past-seven-classes))

(test-case "class-figure refuses a label past 7 as itself"
  (check-exn #rx"^class-figure: label ids run from 0 to 7\n  label: 8"
             (lambda ()
               (class-figure (range 10)
                             (range 10)
                             (list (list "k" past-seven))
                             past-seven-classes
                             #:class-title "x"))))

(test-case "sweep-figure refuses a label past 7 as itself"
  (check-exn #rx"^sweep-figure: label ids run from 0 to 7\n  label: 8"
             (lambda () (sweep-figure (range 10) (range 10) '("k") (list past-seven)))))

(define (mismatches labels classes)
  (count (lambda (cluster class) (not (= cluster class))) labels (renumbered classes)))

(define (span numbers)
  (list (apply min numbers) (apply max numbers)))

(define species (dataset-column 'iris "species"))

(define (flowers)
  (table->device-matrix (load-dataset-table 'iris) #:columns (features 'iris "species")))

(test-gpu "the figures: k-means on iris differs from the species on 16 flowers of shared petal lengths"
  (define labels
    (~> (fit (kmeans #:n-clusters 3) (flowers))
        kmeans-model
        kmeans-result-labels
        device-vector->list))
  (define matched (matched-labels labels species))
  (define truth (renumbered species))
  (define petal-length (dataset-column 'iris "petal-length"))
  (define (petal-lengths keep?)
    (for/list ([petal (in-list petal-length)]
               [cluster (in-list matched)]
               [class (in-list truth)]
               #:when (keep? cluster class))
      petal))
  (define (of-species id)
    (petal-lengths (lambda (_ class) (= class id))))
  (check-equal? (mismatches matched species) 16)
  (check < (apply max (of-species 0)) (apply min (append (of-species 1) (of-species 2))))
  (check-equal? (map span (list (of-species 1) (of-species 2))) '((3.0 5.1) (4.5 6.9)))
  (check-equal? (span (petal-lengths (lambda (cluster class) (not (= cluster class))))) '(4.5 5.1)))

(test-gpu "the figures: wine clusters match the cultivars but for 53 raw and 8 standardized wines, raw ones in bands of proline"
  (define cultivar (dataset-column 'wine "cultivar"))
  (define measurements (features 'wine "cultivar"))
  (define raw-wines (table->device-matrix (load-dataset-table 'wine) #:columns measurements))
  (define (misplaced wines)
    (mismatches (clusters-of (kmeans #:n-clusters 3) wines cultivar) cultivar))
  (check-equal? (remove-duplicates cultivar) '("1" "2" "3"))
  (check-equal? (misplaced raw-wines) 53)
  (check-equal? (misplaced (standardized-matrix 'wine measurements)) 8)
  (define raw (clusters-of (kmeans #:n-clusters 3) raw-wines cultivar))
  (define proline (dataset-column 'wine "proline"))
  (define bands
    (sort (for/list ([cluster (in-list (remove-duplicates raw))])
            (span (for/list ([value (in-list proline)]
                             [label (in-list raw)]
                             #:when (= label cluster))
                    value)))
          <
          #:key first))
  (check-equal? (length bands) 3)
  (check-true (for/and ([band (in-list bands)]
                        [next (in-list (rest bands))])
                (< (second band) (first next)))))

(test-gpu "the figures: DBSCAN's groups keep their numbers across the radii"
  (define eruptions (standardized-matrix 'faithful '("eruptions" "waiting")))
  (define fits
    (for/list ([eps (in-list '(0.1 0.2 0.3 0.5))])
      (dbscan-fit eruptions #:eps eps #:min-samples 5)))
  (define (labels-in result)
    (device-vector->list (dbscan-result-labels result)))
  (define groups (labels-in (third fits)))
  (define matched
    (for/list ([result (in-list fits)])
      (matched-labels (labels-in result) groups)))
  (define core (device-vector->list (dbscan-result-core-sample-indices (third fits))))
  (check-equal? (map (lambda (labels) (sort (remove-duplicates labels) <)) matched)
                '((-1 0 1 2 3 4 5 6 7) (-1 0 1) (-1 0 1) (0)))
  (check-equal? (map (lambda (labels) (count negative? labels)) matched) '(126 25 8 0))
  (check-equal? (for/list ([labels (in-list matched)])
                  (for/list ([group (in-list '(0 1))])
                    (count (lambda (label reference) (and (= label group) (= reference group)))
                           labels
                           (renumbered groups))))
                '((71 42) (160 87) (168 96) (168 0)))
  (check-equal? (count (lambda (label index) (and (>= label 0) (not (memv index core))))
                       (third matched)
                       (range (length groups)))
                12))

(test-gpu "the figures: single linkage keeps versicolor and virginica in one orange cluster"
  (define samples (flowers))
  (define cuts
    (for/list ([k (in-list '(2 3 4 6))])
      (clusters-of (agglomerative-clustering #:n-clusters k) samples species)))
  (check-equal? (map sizes cuts) '((100 50) (98 50 2) (97 50 2 1) (92 50 4 2 1 1)))
  (check-equal? (map (lambda (cut) (count (curry = 1) cut)) cuts) '(100 98 97 92))
  (check-equal? (map (lambda (cut) (sort (remove-duplicates cut) <)) cuts)
                '((0 1) (0 1 2) (0 1 2 3) (0 1 2 3 4 5))))

(test-gpu "the figures: the tumour clusters differ from the diagnosis on 54 and 36 tumours"
  (define diagnosis (dataset-column 'breast-cancer "diagnosis"))
  (define tumours (standardized-matrix 'breast-cancer (features 'breast-cancer "diagnosis" "id")))
  (define radius (dataset-column 'breast-cancer "radius-worst"))
  (define truth (renumbered diagnosis))
  (define (radii id)
    (for/list ([r (in-list radius)]
               [class (in-list truth)]
               #:when (= class id))
      r))
  (match-define (list (list malignant-min malignant-max) (list benign-min benign-max))
    (map (lambda (id) (span (radii id))) '(0 1)))
  (define low (max malignant-min benign-min))
  (define high (min malignant-max benign-max))
  (define clusterings
    (for/list ([estimator (list (kmeans #:n-clusters 2) (spectral-clustering #:n-clusters 2))])
      (clusters-of estimator tumours diagnosis)))
  (define (tumours-where keep? labels)
    (count keep? labels truth radius))
  (check-equal? (remove-duplicates diagnosis) '("M" "B"))
  (check-equal? (map (lambda (labels) (mismatches labels diagnosis)) clusterings) '(54 36))
  (check-equal? (map (curry tumours-where
                            (lambda (label class r) (and (not (= label class)) (<= low r high))))
                     clusterings)
                '(39 32))
  (check-equal? (map (curry tumours-where
                            (lambda (label class r)
                              (and (= label 0) (= class 1) (< r malignant-min))))
                     clusterings)
                '(8 2)))
