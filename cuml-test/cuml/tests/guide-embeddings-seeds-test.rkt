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
         (only-in racket/match match-define struct*)
         (only-in rackunit check check-equal? check-true)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/guide-data.rkt" drawable-label? evaluated-chapter misplaced-by-class))

(struct seeded (seed ari clusters matched trust))

(define umap-ari-floor 0.88)

(define umap-trust-floor 0.985)

(define tsne-trust-floor 0.990)

(define raw-ari-ceiling 0.30)

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
  (define-values (chapter _blocks) (evaluated-chapter "embeddings.scrbl" #:through 'clusterer))
  (define digit (chapter 'digit))
  (define umaps (seeded-fits chapter (umap)))
  (define tsnes (seeded-fits chapter (tsne)))
  (define raw-ari
    (adjusted-rand-score (chapter 'digit-ids) (fit-predict (chapter 'clusterer) (chapter 'images))))
  (for ([fit (in-list umaps)])
    (match-define (struct*
                   seeded
                   ([seed seed] [ari ari] [clusters clusters] [matched matched] [trust trust]))
      fit)
    (define name (format "UMAP seed ~a" seed))
    (check >= ari umap-ari-floor name)
    (check-equal? clusters 10 name)
    (check-true (andmap drawable-label? matched) name)
    (match-define (and misplaced (list _ ones _ _ _ _ _ _ _ nines))
      (misplaced-by-class matched digit))
    (check > (* 2 (+ ones nines)) (apply + misplaced) name)
    (check >= trust umap-trust-floor name))
  (for ([fit (in-list tsnes)])
    (match-define (struct* seeded ([seed seed] [trust trust])) fit)
    (check >= trust tsne-trust-floor (format "t-SNE seed ~a" seed)))
  (check <= raw-ari raw-ari-ceiling)
  (check < (apply max (map seeded-ari tsnes)) (apply min (map seeded-ari umaps))))
