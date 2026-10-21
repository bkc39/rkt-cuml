#lang racket/base

(require cuml
         datasets/core
         pict
         plot/no-gui
         racket/function
         racket/list
         racket/match
         racket/math
         raft
         threading)

(provide agreement
         cluster-colors
         cluster-legend
         cluster-plot
         cluster-style
         crosstab
         dataset-column
         dataset-columns
         dataset-matrix
         features
         labels-of
         line-plot
         matched-labels
         panels
         renumbered
         rows
         sizes
         standardized
         standardized-matrix)

(define (crosstab labels classes)
  (define names (remove-duplicates classes))
  (define (counts cluster)
    (for/list ([name (in-list names)])
      (count (lambda (sample) (equal? (cdr sample) name)) cluster)))
  (~> (map cons labels classes)
      (filter (lambda (sample) (>= (car sample) 0)) _)
      (group-by car _)
      (map counts _)
      (sort < #:key (lambda (row) (index-of row (apply max row))))))

(define (agreement tallies)
  (apply + (map (curry apply max) tallies)))

(define (sizes labels)
  (~> (remove* '(-1) labels) (group-by values _) (map length _) (sort >)))

(define (standardized columns)
  (define (standardize column)
    (define n (length column))
    (define mean (/ (apply + column) n))
    (define sd (sqrt (/ (for/sum ([v (in-list column)]) (sqr (- v mean))) n)))
    (define spread (if (zero? sd) 1 sd))
    (map (lambda (v) (exact->inexact (/ (- v mean) spread))) column))
  (match (memf (negate real?) (append* columns))
    [(cons value _) (raise-argument-error 'standardized "real?" value)]
    [#f (map standardize columns)]))

(define (rows columns)
  (apply map list columns))

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

(define (dataset-matrix name columns #:dtype [type 'float32])
  (~> (dataset-columns name columns) rows (list*->device-matrix #:dtype type)))

(define (standardized-matrix name columns)
  (~> (dataset-columns name columns) standardized rows (list*->device-matrix #:dtype 'float32)))

(define (labels-of estimator samples)
  (device-vector->list (fit-predict estimator samples)))

(define (renumbered labels)
  (define ids
    (for/hash ([label (in-list (remove* '(-1) (remove-duplicates labels)))]
               [id (in-naturals)])
      (values label id)))
  (map (lambda (label) (hash-ref ids label -1)) labels))

(define (matched-labels labels classes)
  (define clusters (remove* '(-1) (remove-duplicates labels)))
  (define names (remove* '(-1) (remove-duplicates classes)))
  (define (shared cluster
            name)
    (count (lambda (label class) (and (equal? label cluster) (equal? class name))) labels classes))
  (define pairs
    (~> (for*/list ([cluster (in-list clusters)]
                    [name (in-list names)])
          (list (shared cluster
                  name)
                cluster
                (index-of names name)))
        (sort > #:key first)))
  (define matches
    (for/fold ([matches (hash)]) ([pair (in-list pairs)])
      (match-define (list n cluster class) pair)
      (cond
        [(zero? n) matches]
        [(hash-has-key? matches cluster) matches]
        [(member class (hash-values matches)) matches]
        [else (hash-set matches cluster class)])))
  (define ids
    (for/fold ([ids matches])
              ([cluster (in-list (filter-not (curry hash-has-key? matches) clusters))]
               [id (in-naturals (length names))])
      (hash-set ids cluster id)))
  (map (lambda (label) (hash-ref ids label -1)) labels))

(define (rgb hex)
  (for/list ([start (in-list '(1 3 5))])
    (string->number (substring hex start (+ start 2)) 16)))

(define cluster-colors
  (map rgb '("#2a78d6" "#eb6834" "#1baf7a" "#eda100" "#e87ba4" "#008300" "#6250d6" "#e34948")))

(define cluster-shapes
  '((fullcircle circle) (fullsquare square)
                        (fulltriangleup triangleup)
                        (fulldiamond diamond)
                        (fulltriangledown triangledown)
                        (full5star 5star)
                        (fulltriangleright triangleright)
                        (full4star 4star)))

(define (cluster-style label)
  (cond
    [(< label (length cluster-colors))
     (cons (list-ref cluster-colors label) (list-ref cluster-shapes label))]
    [else (raise-arguments-error 'cluster-plot "a figure has at most 8 clusters" "label" label)]))

(define noise-color (rgb "#898781"))

(define ink (rgb "#0b0b0b"))

(define (cluster-ids labels)
  (sort (remove* '(-1) (remove-duplicates labels)) <))

(define (cluster-points vs label #:core? [core? #t] #:label [name #f])
  (match-define (list color filled hollow) (cluster-style label))
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

(define (styled-plot renderers xs ys title x-label y-label width height)
  (match-define (list x-min x-max) (padded xs))
  (match-define (list y-min y-max) (padded ys))
  (parameterize ([plot-font-size 10]
                 [plot-foreground (rgb "#5f5e5a")]
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

(define (cluster-renderers xs ys labels core centroids)
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
                        (list (cluster-points (vectors label core?) label)
                              (cluster-points (vectors label (negate core?)) label #:core? #f)))
                      (cluster-ids labels))
          (list (noise-points (vectors -1 (const #t)))
                (centroid-points (map (curry apply vector) centroids)))))

(define (cluster-plot xs
                      ys
                      labels
                      #:core [core #f]
                      #:centroids [centroids '()]
                      #:title [title #f]
                      #:x-label [x-label #f]
                      #:y-label [y-label #f]
                      #:width [width 320]
                      #:height [height 280])
  (match (remove-duplicates (map length (list xs ys labels)))
    [(list _)
     (styled-plot (cluster-renderers xs ys labels core centroids)
                  (append xs (map first centroids))
                  (append ys (map second centroids))
                  title
                  x-label
                  y-label
                  width
                  height)]
    [lengths
     (raise-arguments-error 'cluster-plot "xs, ys and labels differ in length" "lengths" lengths)]))

(define (cluster-legend labels #:names [names #f] #:centroids? [centroids? #f] #:width [width 652])
  (define away (list (vector 2 2)))
  (define entries
    (append (for/list ([label (in-list (cluster-ids labels))])
              (cluster-points away
                              label
                              #:label (if names
                                          (list-ref names label)
                                          (format "cluster ~a" label))))
            (if (memv -1 labels)
                (list (noise-points away #:label "noise"))
                '())
            (if centroids?
                (list (centroid-points away #:label "centroid"))
                '())))
  (define legend-rows (ceiling (/ (length entries) (quotient width 160))))
  (parameterize ([plot-decorations? #f]
                 [plot-foreground (rgb "#5f5e5a")]
                 [plot-legend-anchor 'top]
                 [plot-legend-layout (list 'rows legend-rows 'compact)]
                 [plot-legend-font-size 9])
    (plot-pict entries
               #:x-min 0
               #:x-max 1
               #:y-min 0
               #:y-max 1
               #:width width
               #:height (+ 16 (* 22 legend-rows)))))

(define (line-plot xs
                   ys
                   #:title [title #f]
                   #:x-label [x-label #f]
                   #:y-label [y-label #f]
                   #:width [width 320]
                   #:height [height 240])
  (define vs (map vector xs ys))
  (match-define (cons color _) cluster-colors)
  (parameterize ([plot-x-ticks (ticks-add (plot-x-ticks) xs)])
    (styled-plot (list (lines vs #:color color #:width 2)
                       (points vs #:sym 'fullcircle #:color color #:fill-color color #:size 7))
                 xs
                 ys
                 title
                 x-label
                 y-label
                 width
                 height)))

(define (panels . panel-rows)
  (~> (map (curry apply hc-append 12) panel-rows) (apply vc-append 12 _)))
