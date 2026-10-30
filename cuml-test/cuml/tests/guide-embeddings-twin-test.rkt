#lang racket/base

(require (only-in cuml
                  adjusted-rand-score
                  fit
                  fit-predict
                  trustworthiness
                  tsne
                  tsne-model
                  umap
                  umap-model)
         (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-= check-equal? check-true)
         (only-in raft device-matrix->list* device-vector->list list->device-vector)
         (only-in "private/guide-data.rkt" evaluated-chapter)
         (only-in "private/python-env.rkt" run-twin test-twin)
         (only-in "private/scaler-data.rkt" bits-equal?)
         (only-in "private/tsne-data.rkt" tsne-case)
         (only-in "private/umap-data.rkt" fixture twin-case))

(define-runtime-path twin "python/guide_embeddings_twin.py")

(test-twin "the embeddings chapter's labels, scores and embeddings are Python cuML's"
  (define-values (chapter _blocks) (evaluated-chapter "embeddings.scrbl"))
  (define images (chapter 'images))
  (define rows (device-matrix->list* images))
  (define ids (device-vector->list (chapter 'digit-ids)))
  (define clusterer (chapter 'clusterer))
  (define pictures (list images (chapter 'embedded) (chapter 'pictured)))
  (define labels (map (lambda (samples) (device-vector->list (fit-predict clusterer samples))) pictures))
  (define digits (fixture "digits" rows ids))
  (define (op name . fields)
    (apply hasheq 'op name fields))
  (match-define (hash-table ['results
                             (list* (hash-table ['X sklearn-rows] ['y sklearn-labels])
                                    (hash-table ['embedding umap-embedding])
                                    (hash-table ['embedding tsne-embedding])
                                    found)]
                            ['version version])
    (run-twin
     twin
     (hasheq 'cases
             (append (list (op "digits")
                           (twin-case digits (umap-model (fit (umap) images)))
                           (tsne-case digits (tsne-model (fit (tsne) images))))
                     (for/list ([samples (in-list pictures)])
                       (op "hdbscan" 'X (device-matrix->list* samples)))
                     (for/list ([samples (in-list (cdr pictures))])
                       (op "trust" 'X rows 'E (device-matrix->list* samples)))
                     (for/list ([ours (in-list labels)])
                       (op "ari" 'y ids 'labels ours))))))
  (check-equal? (hash-ref version 'cuml) "26.08.00")
  (check-equal? (hash-ref version 'sklearn) "1.8.0")
  (check-equal? rows sklearn-rows "the pixels are sklearn's digits")
  (check-equal? (chapter 'digit) sklearn-labels "the digits are sklearn's labels")
  (check-true (bits-equal? (device-matrix->list* (chapter 'embedded)) umap-embedding) "UMAP")
  (check-true (bits-equal? (device-matrix->list* (chapter 'pictured)) tsne-embedding) "t-SNE")
  (match-define (list raw-labels umap-labels tsne-labels umap-trust tsne-trust raw-ari umap-ari tsne-ari)
    found)
  (for ([ours (in-list labels)]
        [theirs (in-list (list raw-labels umap-labels tsne-labels))]
        [name (in-list '("pixels" "UMAP" "t-SNE"))])
    (check-equal? ours (hash-ref theirs 'labels) name))
  (for ([samples (in-list (cdr pictures))]
        [theirs (in-list (list umap-trust tsne-trust))])
    (check-equal? (trustworthiness images samples) (hash-ref theirs 't)))
  (for ([ours (in-list labels)]
        [theirs (in-list (list raw-ari umap-ari tsne-ari))])
    (check-= (adjusted-rand-score (chapter 'digit-ids) (list->device-vector ours #:dtype 'int32))
             (hash-ref theirs 'ari)
             1e-12)))
