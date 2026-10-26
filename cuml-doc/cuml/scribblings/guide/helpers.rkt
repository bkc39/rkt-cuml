#lang racket/base

(require cuml
         datasets/core
         pict
         plot/no-gui
         racket/format
         racket/function
         racket/list
         racket/match
         raft
         threading)

(provide class-figure
         class-ids
         cluster-colors
         cluster-legend
         cluster-plot
         cluster-style
         clusters-of
         columns-of
         components-for
         counts-of
         cumulative
         dataset-column
         dataset-columns
         features
         labels-of
         leading
         line-plot
         matched-labels
         renumbered
         rounded
         rows
         rows->matrix
         sizes
         spectrum
         sweep-figure)

(define (sizes labels)
  (~> (remove* '(-1) labels) (group-by values _) (map length _) (sort >)))

(define (rows columns)
  (apply map list columns))

(define (rows->matrix samples)
  (list*->device-matrix samples #:dtype 'float32))

(define (dataset-columns name columns)
  (~> (load-dataset-table name #:columns columns)
      dataset-table-columns
      vector->list
      (map vector->list _)))

(define (dataset-column name column)
  (match-define (list column-values) (dataset-columns name (list column)))
  column-values)

(define (features name . excluded)
  (~> (load-dataset-table name) dataset-table-names (remove* excluded _)))

(define (labels-of estimator samples)
  (device-vector->list (fit-predict estimator samples)))

(define (distinct-clusters labels)
  (remove* '(-1) (remove-duplicates labels)))

(define (relabelled labels ids)
  (map (lambda (label) (hash-ref ids label -1)) labels))

(define (renumbered labels)
  (relabelled labels
              (for/hash ([label (in-list (distinct-clusters labels))]
                         [id (in-naturals)])
                (values label id))))

(define (class-ids classes)
  (~> classes renumbered (list->device-vector #:dtype 'int32)))

(define (counts-of ids labels)
  (~> (contingency-matrix ids labels) contingency-table-counts device-matrix->list*))

(define (rounded x)
  (~r x #:precision 3))

(define (matched-labels labels classes)
  (define clusters (distinct-clusters labels))
  (define names (distinct-clusters classes))
  (define (overlap cluster name)
    (count (lambda (label class) (equal? (cons label class) (cons cluster name))) labels classes))
  (define pairs
    (~> (for*/list ([cluster (in-list clusters)]
                    [name (in-list names)])
          (list (overlap cluster name) cluster (index-of names name)))
        (sort > #:key first)))
  (define matches
    (for/fold ([matches (hash)]) ([pair (in-list pairs)])
      (match-define (list n cluster class) pair)
      (cond
        [(or (zero? n) (hash-has-key? matches cluster) (member class (hash-values matches))) matches]
        [else (hash-set matches cluster class)])))
  (define ids
    (for/fold ([ids matches])
              ([cluster (in-list (filter-not (curry hash-has-key? matches) clusters))]
               [id (in-naturals (length names))])
      (hash-set ids cluster id)))
  (relabelled labels ids))

(define (clusters-of estimator samples classes)
  (matched-labels (labels-of estimator samples) classes))

(define (largest-entries entries names n)
  (~> (map cons entries names) (sort > #:key (compose abs car)) (take n) (map cdr _)))

(define (leading result names n)
  (~> result pca-result-components device-matrix->list* (map (curryr largest-entries names n) _)))

(define (cumulative shares)
  (for/fold ([sums '()]
             [total 0]
             #:result (reverse sums))
            ([share (in-list shares)])
    (define sum (+ total share))
    (values (cons sum sums) sum)))

(define (components-for shares share)
  (~> (cumulative shares) (index-where (curry <= share)) add1))

(define (columns-of matrix)
  (~> matrix device-matrix->list* (apply map list _)))

(define (spectrum result n)
  (~> (match result
        [(? pca-result?)
         (list (pca-result-singular-values result) (pca-result-explained-variance-ratio result))]
        [(? tsvd-result?)
         (list (tsvd-result-singular-values result) (tsvd-result-explained-variance-ratio result))])
      (map (lambda (entries) (~> entries device-vector->list (take n) (map rounded _))) _)))

(define (rgb hex)
  (for/list ([start (in-list '(1 3 5))])
    (string->number (substring hex start (+ start 2)) 16)))

(define cluster-colors
  (map rgb '("#2a78d6" "#eb6834" "#1baf7a" "#eda100" "#e87ba4" "#008300" "#6250d6" "#e34948")))

(define cluster-shapes
  (list '(fullcircle circle)
        '(fullsquare square)
        '(fulltriangleup triangleup)
        '(fulldiamond diamond)
        '(fulltriangledown triangledown)
        '(full5star 5star)
        '(fulltriangleright triangleright)
        '(full4star 4star)))

(define (cluster-style label [who 'cluster-style])
  (cond
    [(and (exact-nonnegative-integer? label) (< label (length cluster-colors)))
     (cons (list-ref cluster-colors label) (list-ref cluster-shapes label))]
    [else (raise-arguments-error who "label ids run from 0 to 7" "label" label)]))

(define noise-color (rgb "#898781"))

(define axis-color (rgb "#5f5e5a"))

(define ink (rgb "#0b0b0b"))

(define line-ink (first cluster-colors))

(define (cluster-ids labels)
  (sort (distinct-clusters labels) <))

(define (cluster-points vs label who #:core? [core? #t] #:label [name #f])
  (match-define (list color filled hollow) (cluster-style label who))
  (points vs
          #:sym (if core? filled hollow)
          #:color color
          #:fill-color color
          #:size (if core? 7 5)
          #:alpha 0.8
          #:label name))

(define (noise-points vs #:label [name #f])
  (points vs #:sym 'circle #:color noise-color #:size 4 #:label name))

(define (centroid-points vs #:label [name #f])
  (points vs #:sym 'times #:color ink #:size 14 #:line-width 3 #:label name))

(define (padded numbers)
  (define low (apply min numbers))
  (define high (apply max numbers))
  (define pad
    (if (= low high)
        1/2
        (* 1/20 (- high low))))
  (list (- low pad) (+ high pad)))

(define (styled-plot renderers
                     xs
                     ys
                     #:title title
                     #:x-label x-label
                     #:y-label y-label
                     #:width width
                     #:height height)
  (match-define (list x-min x-max) (padded xs))
  (match-define (list y-min y-max) (padded ys))
  (parameterize ([plot-font-size 10]
                 [plot-foreground axis-color]
                 [plot-tick-size 6])
    (plot-pict renderers
               #:x-min x-min
               #:x-max x-max
               #:y-min y-min
               #:y-max y-max
               #:title title
               #:x-label x-label
               #:y-label y-label
               #:width width
               #:height height)))

(define (cluster-renderers xs ys labels #:core core #:centroids centroids #:who who)
  (define core?
    (if core
        (lambda (index) (memv index core))
        (const #t)))
  (define (vectors label keep?)
    (for/list ([x (in-list xs)]
               [y (in-list ys)]
               [sample-label (in-list labels)]
               [index (in-naturals)]
               #:when (and (= sample-label label) (keep? index)))
      (vector x y)))
  (append (append-map (lambda (label)
                        (list (cluster-points (vectors label core?) label who)
                              (cluster-points (vectors label (negate core?)) label who #:core? #f)))
                      (cluster-ids labels))
          (list (noise-points (vectors -1 (const #t)))
                (centroid-points (map list->vector centroids)))))

(define (cluster-plot xs
                      ys
                      labels
                      #:core [core #f]
                      #:centroids [centroids '()]
                      #:title [title #f]
                      #:x-label [x-label #f]
                      #:y-label [y-label #f]
                      #:width [width 320]
                      #:height [height 280]
                      #:who [who 'cluster-plot])
  (match (remove-duplicates (map length (list xs ys labels)))
    [(list _)
     (styled-plot (cluster-renderers xs ys labels #:core core #:centroids centroids #:who who)
                  (append xs (map first centroids))
                  (append ys (map second centroids))
                  #:title title
                  #:x-label x-label
                  #:y-label y-label
                  #:width width
                  #:height height)]
    [lengths (raise-arguments-error who "xs, ys and labels differ in length" "lengths" lengths)]))

(define legend-entry-width 160)

(define legend-margin 16)

(define legend-row-height 22)

(define (cluster-legend labels #:names [names '()] #:centroids? [centroids? #f] #:width [width 652])
  (define away (list (vector 2 2)))
  (define (name-of label)
    (cond
      [(and (exact-nonnegative-integer? label) (< label (length names))) (list-ref names label)]
      [else (format "cluster ~a" label)]))
  (define cluster-entries
    (for/list ([label (in-list (cluster-ids labels))])
      (cluster-points away label 'cluster-legend #:label (name-of label))))
  (define noise-entry (and (memv -1 labels) (noise-points away #:label "noise")))
  (define centroid-entry (and centroids? (centroid-points away #:label "centroid")))
  (define entries (append cluster-entries (filter values (list noise-entry centroid-entry))))
  (define legend-rows (ceiling (/ (length entries) (max 1 (quotient width legend-entry-width)))))
  (parameterize ([plot-decorations? #f]
                 [plot-foreground axis-color]
                 [plot-legend-anchor 'top]
                 [plot-legend-layout `(rows ,legend-rows compact)]
                 [plot-legend-font-size 9])
    (plot-pict entries
               #:x-min 0
               #:x-max 1
               #:y-min 0
               #:y-max 1
               #:width width
               #:height (+ legend-margin (* legend-row-height legend-rows)))))

(define (paired plots)
  (match plots
    [(list* left right more) (cons (list left right) (paired more))]
    [(list alone) (list (list alone))]
    ['() '()]))

(define (figure plots legend)
  (~> (paired plots)
      (map (curry apply hc-append 12) _)
      (apply vc-append 12 _)
      (vc-append 12 _ legend)))

(define (centroids-of clustering)
  (match clustering
    [(list _ _) '()]
    [(list _ _ centroids) centroids]))

(define (class-names classes title)
  (for/list ([class (in-list (distinct-clusters classes))])
    (if (number? class)
        (format "~a ~a" title class)
        (~a class))))

(define (class-figure #:class-title class-title
                      #:names [names #f]
                      #:x-label [x-label #f]
                      #:y-label [y-label #f]
                      . arguments)
  (match-define (list xs ys clusterings classes)
    (match arguments
      [(list (? device-matrix? samples) clusterings classes)
       (append (columns-of samples) (list clusterings classes))]
      [_ arguments]))
  (define truth (renumbered classes))
  (define plots
    (for/list ([clustering (in-list (append clusterings (list (list class-title truth))))])
      (match-define (list* title labels _) clustering)
      (cluster-plot xs
                    ys
                    labels
                    #:centroids (centroids-of clustering)
                    #:title title
                    #:x-label x-label
                    #:y-label y-label
                    #:who 'class-figure)))
  (define centroids? (ormap pair? (map centroids-of clusterings)))
  (figure plots
          (cluster-legend (append truth (append-map second clusterings))
                          #:names (or names (class-names classes class-title))
                          #:centroids? centroids?)))

(define (sweep-figure xs
                      ys
                      titles
                      clusterings
                      #:cores [cores (map (const #f) clusterings)]
                      #:names [names '()]
                      #:x-label [x-label #f]
                      #:y-label [y-label #f])
  (define (panel title labels core)
    (cluster-plot xs
                  ys
                  labels
                  #:core core
                  #:title title
                  #:x-label x-label
                  #:y-label y-label
                  #:height 260
                  #:who 'sweep-figure))
  (match (remove-duplicates (map length (list titles clusterings cores)))
    [(list _)
     (figure (map panel titles clusterings cores)
             (cluster-legend (append* clusterings) #:names names))]
    [lengths
     (raise-arguments-error 'sweep-figure
                            "titles, clusterings and cores differ in length"
                            "lengths"
                            lengths)]))

(define (line-plot xs
                   ys
                   #:title [title #f]
                   #:x-label [x-label #f]
                   #:y-label [y-label #f]
                   #:width [width 320]
                   #:height [height 240])
  (define vs (map vector xs ys))
  (parameterize ([plot-x-ticks (ticks-add (plot-x-ticks) xs)])
    (styled-plot (list (lines vs #:color line-ink #:width 2)
                       (points vs #:sym 'fullcircle #:color line-ink #:size 7))
                 xs
                 ys
                 #:title title
                 #:x-label x-label
                 #:y-label y-label
                 #:width width
                 #:height height)))
