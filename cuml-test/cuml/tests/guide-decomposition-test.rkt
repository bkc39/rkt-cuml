#lang racket/base

(require (only-in cuml
                  adjusted-rand-score
                  estimator-set-params
                  fit
                  fit-predict
                  kmeans-model
                  kmeans-result-centroids
                  kmeans-result-labels
                  kmeans-transform)
         (only-in cuml/scribblings/guide/helpers clusters-of components-for)
         (only-in racket/function curry)
         (only-in racket/list last range)
         (only-in racket/match match match-define)
         (only-in racket/promise delay force)
         (only-in rackunit check check-equal? check-regexp-match test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/guide-data.rkt" check-scored evaluated-chapter flattened-text mismatches))

(define decomposition
  (delay
    (evaluated-chapter "decomposition.scrbl")))

(define (centroid-gaps fitted samples)
  (~> (kmeans-transform (~> fitted kmeans-model kmeans-result-centroids) samples)
      device-matrix->list*
      (map (lambda (distances)
             (match (sort (map sqrt distances) <)
               [(list* nearest next _) (- next nearest)]))
           _)))

(define scored
  (list '(scaled ((0 3 48) (0 65 0) (59 3 0)) "0.897")
        '(projected ((0 1 48) (0 65 0) (59 5 0)) "0.895")
        '(whitened ((0 1 48) (1 67 0) (58 3 0)) "0.912")))

(define least-gaps '((projected 0.071) (whitened 0.126)))

(test-gpu "decomposition: the chapter's k-means scores"
  (define-values (chapter blocks) (force decomposition))
  (match-define (list* _ _ _ k-means _) blocks)
  (check-equal? (last k-means) '("0.897" "0.895" "0.912"))
  (for ([row (in-list scored)])
    (match-define (list name columns text) row)
    (check-scored (chapter 'cultivar-ids)
                  (fit-predict (chapter 'clusterer) (chapter name))
                  columns
                  text)))

(test-gpu "decomposition: centroid margins on two components"
  (define-values (chapter _blocks) (force decomposition))
  (define clusterer (chapter 'clusterer))
  (define all-labels (fit-predict clusterer (chapter 'scaled)))
  (for ([row (in-list least-gaps)])
    (match-define (list name least-gap) row)
    (define reduced (chapter name))
    (define labels (fit-predict clusterer reduced))
    (define fitted (fit clusterer reduced))
    (check >= (adjusted-rand-score all-labels labels) 0.85 (symbol->string name))
    (check-equal? (~> fitted kmeans-model kmeans-result-labels device-vector->list)
                  (device-vector->list labels)
                  (symbol->string name))
    (check >= (apply min (centroid-gaps fitted reduced)) least-gap (symbol->string name))))

(test-gpu "decomposition: misplaced wines, seeds 0 to 9 and 73"
  (define-values (chapter _blocks) (force decomposition))
  (define cultivar (chapter 'cultivar))
  (define samples (map chapter '(scaled projected whitened)))
  (for ([seed (in-list (append (range 10) '(73)))])
    (define clusterer (estimator-set-params (chapter 'clusterer) #:random-state seed))
    (match-define (list all projected whitened)
      (map (lambda (reduced) (mismatches (clusters-of clusterer reduced cultivar) cultivar)) samples))
    (check <= whitened (min all projected) (format "seed ~a" seed))
    (check <= (abs (- projected all)) 1 (format "seed ~a" seed))))

(test-case "components-for and its description"
  (check-equal? (map (curry components-for '(1/2 3/10 1/5)) '(1/4 4/5 1)) '(1 2 3))
  (check-regexp-match
   #rx"@racket\\[components-for\\] gives the fewest leading components whose cumulative share reaches a share"
   (flattened-text (collection-file-path "examples.scrbl" "cuml" "scribblings" "guide"))))
