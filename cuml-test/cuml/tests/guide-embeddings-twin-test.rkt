#lang racket/base

(require (only-in cuml
                  adjusted-rand-score
                  fit
                  fit-predict
                  trustworthiness
                  tsne
                  tsne-model
                  tsne-result-embedding
                  umap
                  umap-model)
         (only-in racket/match match-define)
         (only-in racket/promise delay force)
         racket/runtime-path
         (only-in rackunit check-= check-equal? check-true)
         (only-in raft device-matrix->list* device-vector->list list->device-vector)
         (only-in threading ~>)
         (only-in "private/guide-data.rkt" evaluated-chapter)
         (only-in "private/python-env.rkt" run-twin test-twin)
         (only-in "private/scaler-data.rkt" bits-equal?)
         (only-in "private/tsne-data.rkt" tsne-case)
         (only-in "private/umap-data.rkt" fixture twin-case))

(define-runtime-path script "python/guide_embeddings_twin.py")

(define pictures '(images embedded pictured))

(define embedded-pictures (cdr pictures))

(struct twin-run (chapter labels tsne-seed-1 theirs version))

(define (op name . fields)
  (apply hasheq 'op name fields))

(define twin
  (delay
    (define-values (chapter _blocks) (evaluated-chapter "embeddings.scrbl"))
    (define images (chapter 'images))
    (define rows (device-matrix->list* images))
    (define ids (device-vector->list (chapter 'digit-ids)))
    (define digits (fixture "digits" rows ids))
    (define tsne-seed-1 (tsne-model (fit (tsne #:random-state 1) images)))
    (define labels
      (for/hasheq ([name (in-list pictures)])
        (values name (device-vector->list (fit-predict (chapter 'clusterer) (chapter name))))))
    (define (samples name)
      (device-matrix->list* (chapter name)))
    (define cases
      (append (list (cons 'digits (op "digits"))
                    (cons 'umap (twin-case digits (umap-model (fit (umap) images))))
                    (cons 'tsne (tsne-case digits (tsne-model (fit (tsne) images))))
                    (cons 'tsne-seed-1 (tsne-case digits tsne-seed-1)))
              (for/list ([name (in-list pictures)])
                (cons (list 'hdbscan name) (op "hdbscan" 'X (samples name))))
              (for/list ([name (in-list embedded-pictures)])
                (cons (list 'trust name) (op "trust" 'X rows 'E (samples name))))
              (for/list ([name (in-list pictures)])
                (cons (list 'ari name) (op "ari" 'y ids 'labels (hash-ref labels name))))))
    (match-define (hash-table ['results results] ['version version])
      (run-twin script (hasheq 'cases (map cdr cases))))
    (twin-run chapter labels tsne-seed-1 (make-immutable-hash (map cons (map car cases) results)) version)))

(define (theirs key field)
  (hash-ref (hash-ref (twin-run-theirs (force twin)) key) field))

(test-twin "the embeddings twin runs Python cuML 26.08.00 and sklearn 1.8.0"
  (define version (twin-run-version (force twin)))
  (check-equal? (hash-ref version 'cuml) "26.08.00")
  (check-equal? (hash-ref version 'sklearn) "1.8.0"))

(test-twin "the embeddings chapter's pixels and digits are sklearn's"
  (define chapter (twin-run-chapter (force twin)))
  (check-equal? (device-matrix->list* (chapter 'images)) (theirs 'digits 'X))
  (check-equal? (chapter 'digit) (theirs 'digits 'y)))

(test-twin "the embeddings chapter's UMAP and t-SNE embeddings are Python cuML's, bitwise"
  (define chapter (twin-run-chapter (force twin)))
  (check-true (bits-equal? (device-matrix->list* (chapter 'embedded)) (theirs 'umap 'embedding))
              "UMAP")
  (check-true (bits-equal? (device-matrix->list* (chapter 'pictured)) (theirs 'tsne 'embedding))
              "t-SNE"))

(test-twin "t-SNE at seed 1 on the digits is Python cuML's at seed 1, bitwise"
  (check-true (bits-equal? (~> (force twin) twin-run-tsne-seed-1 tsne-result-embedding device-matrix->list*)
                           (theirs 'tsne-seed-1 'embedding))))

(test-twin "HDBSCAN labels the pixels and both embeddings as Python cuML does"
  (define labels (twin-run-labels (force twin)))
  (for ([name (in-list pictures)])
    (check-equal? (hash-ref labels name) (theirs (list 'hdbscan name) 'labels) (symbol->string name))))

(test-twin "the embeddings' trustworthiness is Python cuML's"
  (define chapter (twin-run-chapter (force twin)))
  (for ([name (in-list embedded-pictures)])
    (check-equal? (trustworthiness (chapter 'images) (chapter name)) (theirs (list 'trust name) 't)
                  (symbol->string name))))

(test-twin "the clusterings' adjusted Rand indices are sklearn's"
  (match-define (twin-run chapter labels _ _ _) (force twin))
  (for ([name (in-list pictures)])
    (check-= (adjusted-rand-score (chapter 'digit-ids)
                                  (list->device-vector (hash-ref labels name) #:dtype 'int32))
             (theirs (list 'ari name) 'ari)
             1e-12
             (symbol->string name))))
