#lang racket/base

(require (only-in cuml
                  adjusted-rand-score
                  estimator-set-params
                  fit-predict
                  fit-transform
                  trustworthiness
                  tsne
                  umap)
         (only-in cuml/scribblings/guide/helpers matched-labels sizes)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match-define)
         (only-in rackunit check check-equal? check-true)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/guide-data.rkt" evaluated-chapter misplaced-by-class))

(struct seeded (seed ari clusters matched trust))

(define (seeded-fits chapter estimator)
  (define images (chapter 'images))
  (define clusterer (chapter 'clusterer))
  (for/list ([seed (in-range 10)])
    (define samples (fit-transform (estimator-set-params estimator #:random-state seed) images))
    (define labels (fit-predict clusterer samples))
    (define found (device-vector->list labels))
    (seeded seed
            (adjusted-rand-score (chapter 'digit-ids) labels)
            (length (sizes found))
            (matched-labels found (chapter 'digit))
            (trustworthiness images samples))))

(test-gpu "embeddings, seeds 0 to 9: UMAP clusters the digits, t-SNE less well, the pixels worst"
  (define-values (chapter _blocks) (evaluated-chapter "embeddings.scrbl"))
  (define digit (chapter 'digit))
  (define umaps (seeded-fits chapter (umap)))
  (define tsnes (seeded-fits chapter (tsne)))
  (define raw-ari
    (adjusted-rand-score (chapter 'digit-ids) (fit-predict (chapter 'clusterer) (chapter 'images))))
  (for ([fit (in-list umaps)])
    (match-define (seeded seed ari clusters matched trust) fit)
    (define name (format "UMAP seed ~a" seed))
    (check >= ari 0.88 name)
    (check-equal? clusters 10 name)
    (check-true (for/and ([id (in-list (remove-duplicates matched))])
                  (<= -1 id 9))
                name)
    (define misplaced (misplaced-by-class matched digit))
    (match-define (list _ ones _ _ _ _ _ _ _ nines) misplaced)
    (check > (* 2 (+ ones nines)) (apply + misplaced) name)
    (check >= trust 0.985 name))
  (for ([fit (in-list tsnes)])
    (check >= (seeded-trust fit) 0.990 (format "t-SNE seed ~a" (seeded-seed fit))))
  (check <= raw-ari 0.30)
  (check < (apply max (map seeded-ari tsnes)) (apply min (map seeded-ari umaps))))
