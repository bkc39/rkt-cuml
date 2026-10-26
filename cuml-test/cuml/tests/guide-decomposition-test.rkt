#lang racket/base

(require (only-in cuml
                  adjusted-rand-score
                  estimator-set-params
                  fit
                  fit-predict
                  fit-transform
                  kmeans-model
                  kmeans-result-centroids
                  kmeans-result-labels
                  kmeans-transform
                  pca)
         (only-in cuml/scribblings/guide/helpers
                  class-ids
                  clusters-of
                  components-for
                  dataset-column
                  features)
         (only-in racket/function curry)
         (only-in racket/list range)
         (only-in racket/match match match-define)
         (only-in racket/string string-contains?)
         (only-in rackunit check check-equal? check-not-false test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/guide-data.rkt"
                  chapter-clusterer
                  check-scored
                  flattened-text
                  mismatches
                  scaled-matrix))

(define cultivar (dataset-column 'wine "cultivar"))

(define (cultivar-ids)
  (class-ids cultivar))

(define (centroid-gaps fitted samples)
  (~> (kmeans-transform (~> fitted kmeans-model kmeans-result-centroids) samples)
      device-matrix->list*
      (map (lambda (distances)
             (match (sort (map sqrt distances) <)
               [(list* nearest next _) (- next nearest)]))
           _)))

(define (projections)
  (define scaled (scaled-matrix 'wine (features 'wine "cultivar")))
  (list scaled
        (fit-transform (pca #:n-components 2) scaled)
        (fit-transform (pca #:n-components 2 #:whiten #t) scaled)))

(test-gpu "the guide: decomposition, the chapter's k-means on two components clusters about as well as on 13 columns, whitened better"
  (define clusterer (chapter-clusterer))
  (define samples (projections))
  (match-define (list all-labels projected-labels whitened-labels)
    (map (curry fit-predict clusterer) samples))
  (check-scored (cultivar-ids) all-labels '((0 3 48) (0 65 0) (59 3 0)) "0.897")
  (check-scored (cultivar-ids) projected-labels '((0 1 48) (0 65 0) (59 5 0)) "0.895")
  (check-scored (cultivar-ids) whitened-labels '((0 1 48) (1 67 0) (58 3 0)) "0.912")
  (for ([labels (in-list (list projected-labels whitened-labels))]
        [reduced (in-list (cdr samples))]
        [least-gap (in-list '(0.071 0.126))])
    (check >= (adjusted-rand-score all-labels labels) 0.85)
    (define fitted (fit clusterer reduced))
    (check-equal? (~> fitted kmeans-model kmeans-result-labels device-vector->list)
                  (device-vector->list labels))
    (check >= (apply min (centroid-gaps fitted reduced)) least-gap)))

(test-gpu "the guide: decomposition, whitened misplaces the fewest wines and two components within one wine of 13 columns, seeds 0 to 9 and 73"
  (define samples (projections))
  (for ([seed (in-list (append (range 10) '(73)))])
    (define clusterer (estimator-set-params (chapter-clusterer) #:random-state seed))
    (match-define (list all projected whitened)
      (map (lambda (reduced) (mismatches (clusters-of clusterer reduced cultivar) cultivar))
           samples))
    (check <= whitened (min all projected) (format "seed ~a" seed))
    (check <= (abs (- projected all)) 1 (format "seed ~a" seed))))

(test-case "the guide: components-for gives the fewest leading components whose cumulative share reaches a share"
  (check-equal? (map (curry components-for '(1/2 3/10 1/5)) '(1/4 4/5 1)) '(1 2 3))
  (check-not-false
   (string-contains? (flattened-text (collection-file-path "examples.scrbl" "cuml" "scribblings" "guide"))
                     (string-append "@racket[components-for] gives the fewest leading components"
                                    " whose cumulative share reaches a share"))))
