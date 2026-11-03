#lang racket/base

(require (only-in cuml
                  adjusted-rand-score
                  fit
                  fit-predict
                  hdbscan
                  trustworthiness
                  tsne
                  tsne-model
                  tsne-result-embedding
                  umap
                  umap-model)
         (only-in racket/match match-define struct*)
         (only-in racket/promise delay force)
         racket/runtime-path
         (only-in rackunit check-= check-equal? check-true)
         (only-in raft device-matrix->list* device-vector->list list->device-vector)
         (only-in threading ~>)
         (only-in "private/guide-data.rkt" embeddings-pin evaluated-chapter)
         (only-in "private/python-env.rkt" run-twin test-twin)
         (only-in "private/scaler-data.rkt" bits-equal?)
         (only-in "private/tsne-data.rkt" tsne-case)
         (only-in "private/umap-data.rkt" fixture twin-case))

(define-runtime-path script "python/guide_embeddings_twin.py")

(define pictures '(images embedded pictured))

(define embedded-pictures (cdr pictures))

(struct twin-run (chapter labels coarse tsne-seed-1 theirs version))

(define (op name . fields)
  (apply hasheq 'op name fields))

(define coarse-cluster-size 50)

(define coarse-min-samples 5)

(define (hdbscan-op samples min-cluster-size min-samples)
  (op "hdbscan" 'X samples 'min_cluster_size min-cluster-size 'min_samples min-samples))

(define twin
  (delay
    (define-values (chapter _blocks) (evaluated-chapter "embeddings.scrbl" #:through 'pictured))
    (define images (chapter 'images))
    (define rows (device-matrix->list* images))
    (define ids (device-vector->list (chapter 'digit-ids)))
    (define digits (fixture "digits" rows ids))
    (define tsne-seed-1 (tsne-model (fit (tsne #:random-state 1) images)))
    (define coarse
      (~> (hdbscan #:min-cluster-size coarse-cluster-size #:min-samples coarse-min-samples)
          (fit-predict (chapter 'embedded))
          device-vector->list))
    (define labels
      (for/hasheq ([name (in-list pictures)])
        (values name (device-vector->list (fit-predict (chapter 'clusterer) (chapter name))))))
    (define (samples name)
      (device-matrix->list* (chapter name)))
    (define cases
      (append (list (cons 'digits (op "digits"))
                    (cons 'umap (twin-case digits (umap-model (fit (umap) images))))
                    (cons 'tsne (tsne-case digits (tsne-model (fit (tsne) images))))
                    (cons 'tsne-seed-1 (tsne-case digits tsne-seed-1))
                    (cons 'coarse
                          (hdbscan-op (samples 'embedded) coarse-cluster-size coarse-min-samples)))
              (for/list ([name (in-list pictures)])
                (cons (list 'hdbscan name)
                      (hdbscan-op (samples name)
                                  (embeddings-pin 'min-cluster-size)
                                  (embeddings-pin 'min-samples))))
              (for/list ([name (in-list embedded-pictures)])
                (cons (list 'trust name) (op "trust" 'X rows 'E (samples name))))
              (for/list ([name (in-list pictures)])
                (cons (list 'ari name) (op "ari" 'y ids 'labels (hash-ref labels name))))))
    (match-define (hash-table ['results results] ['version version])
      (run-twin script (hasheq 'cases (map cdr cases))))
    (twin-run chapter
              labels
              coarse
              tsne-seed-1
              (make-immutable-hash (map cons (map car cases) results))
              version)))

(define (theirs key field)
  (match-define (struct* twin-run ([theirs results])) (force twin))
  (hash-ref (hash-ref results key) field))

(test-twin "the embeddings twin runs Python cuML 26.08.00 and sklearn 1.8.0"
  (match-define (struct* twin-run ([version version])) (force twin))
  (check-equal? (hash-ref version 'cuml) "26.08.00")
  (check-equal? (hash-ref version 'sklearn) "1.8.0"))

(test-twin "the embeddings chapter's pixels and digits are sklearn's"
  (match-define (struct* twin-run ([chapter chapter])) (force twin))
  (check-equal? (device-matrix->list* (chapter 'images)) (theirs 'digits 'X))
  (check-equal? (chapter 'digit) (theirs 'digits 'y)))

(test-twin "the embeddings chapter's UMAP and t-SNE embeddings are Python cuML's, bitwise"
  (match-define (struct* twin-run ([chapter chapter])) (force twin))
  (check-true (bits-equal? (device-matrix->list* (chapter 'embedded)) (theirs 'umap 'embedding))
              "UMAP")
  (check-true (bits-equal? (device-matrix->list* (chapter 'pictured)) (theirs 'tsne 'embedding))
              "t-SNE"))

(test-twin "t-SNE at seed 1 on the digits is Python cuML's at seed 1, bitwise"
  (match-define (struct* twin-run ([tsne-seed-1 fitted])) (force twin))
  (check-true (bits-equal? (~> fitted tsne-result-embedding device-matrix->list*)
                           (theirs 'tsne-seed-1 'embedding))))

(test-twin "HDBSCAN labels the pixels and both embeddings as Python cuML does"
  (match-define (struct* twin-run ([labels labels])) (force twin))
  (for ([name (in-list pictures)])
    (check-equal? (hash-ref labels name)
                  (theirs (list 'hdbscan name) 'labels)
                  (symbol->string name))))

(test-twin "HDBSCAN at 50 digits a cluster and 5 neighbours labels UMAP's embedding as Python cuML does"
  (match-define (struct* twin-run ([coarse coarse])) (force twin))
  (check-equal? coarse (theirs 'coarse 'labels)))

(test-twin "the embeddings' trustworthiness is Python cuML's"
  (match-define (struct* twin-run ([chapter chapter])) (force twin))
  (for ([name (in-list embedded-pictures)])
    (check-equal? (trustworthiness (chapter 'images) (chapter name))
                  (theirs (list 'trust name) 't)
                  (symbol->string name))))

(test-twin "the clusterings' adjusted Rand indices are sklearn's"
  (match-define (struct* twin-run ([chapter chapter] [labels labels])) (force twin))
  (for ([name (in-list pictures)])
    (check-= (adjusted-rand-score (chapter 'digit-ids)
                                  (list->device-vector (hash-ref labels name) #:dtype 'int32))
             (theirs (list 'ari name) 'ari)
             1e-12
             (symbol->string name))))
