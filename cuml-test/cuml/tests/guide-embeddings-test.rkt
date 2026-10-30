#lang racket/base

(require (only-in cuml fit-predict fit-transform pca trustworthiness tsne umap)
         (only-in cuml/scribblings/guide/helpers clusters-of rounded)
         (only-in pict pict?)
         (only-in racket/list filter-not last range remove-duplicates)
         (only-in racket/match match-define)
         (only-in racket/promise delay force)
         (only-in rackunit check check-equal? check-not-false)
         (only-in raft device-matrix->list*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/guide-data.rkt"
                  check-scored
                  cluster-columns
                  evaluated-chapter
                  misplaced-by-class
                  mismatches))

(define embeddings
  (delay
    (evaluated-chapter "embeddings.scrbl")))

(define raw-table
  '((4 174 0 0 0 0 0) (67 0 0 0 0 0 115)
                      (75 0 0 0 102 0 0)
                      (65 0 0 0 0 0 118)
                      (48 0 0 133 0 0 0)
                      (88 0 0 0 0 0 94)
                      (20 0 0 0 0 160 1)
                      (45 0 134 0 0 0 0)
                      (128 0 0 0 0 0 46)
                      (93 0 0 0 0 0 87)))

(define umap-table
  '((0 0 178 0 0 0 0 0 0 0 0) (1 0 0 27 0 0 0 0 0 154 0)
                              (5 0 0 0 166 0 0 0 1 0 5)
                              (0 0 0 0 0 0 2 0 181 0 0)
                              (0 177 0 0 0 0 4 0 0 0 0)
                              (0 0 0 1 0 179 0 2 0 0 0)
                              (0 0 0 180 0 0 0 0 0 0 1)
                              (0 0 0 0 0 0 179 0 0 0 0)
                              (0 0 0 0 0 0 0 0 0 4 170)
                              (0 0 0 0 0 2 11 142 3 20 2)))

(define tsne-table
  '((6 0 0 172 0 0 0 0 0 0 0) (13 27 0 0 0 0 0 0 0 53 89)
                              (11 166 0 0 0 0 0 0 0 0 0)
                              (32 0 0 0 0 0 0 0 151 0 0)
                              (8 0 0 0 173 0 0 0 0 0 0)
                              (19 0 1 0 0 0 0 162 0 0 0)
                              (2 0 179 0 0 0 0 0 0 0 0)
                              (9 0 0 0 0 170 0 0 0 0 0)
                              (46 0 0 0 0 0 0 0 0 1 127)
                              (41 0 0 0 0 7 114 0 0 18 0)))

(define (printed blocks)
  (map (lambda (forms) (filter-not pict? forms)) blocks))

(test-gpu "embeddings: the chapter prints its sizes, scores and trustworthiness"
  (define-values (_chapter blocks) (force embeddings))
  (match-define (list data raw umap-block _ (list _ tsne-scored trust _)) blocks)
  (check-equal? (last data) '(64 (183 182 182 181 181 180 179 178 177 174)))
  (check-equal? (last raw) '("0.284" (461 174 160 134 133 102) 633))
  (check-equal? (last umap-block) '("0.899" (208 196 185 181 178 178 178 177 166 144) 6))
  (check-equal? tsne-scored '("0.745" (216 193 180 177 173 172 162 151 114 72) 187))
  (check-equal? trust '("0.83" "0.988" "0.992")))

(test-gpu "embeddings: the clusters' tables on the pixels, UMAP and t-SNE"
  (define-values (chapter _blocks) (force embeddings))
  (define ids (chapter 'digit-ids))
  (define clusterer (chapter 'clusterer))
  (for ([name (in-list '(images embedded pictured))]
        [table (in-list (list raw-table umap-table tsne-table))]
        [text (in-list '("0.284" "0.899" "0.745"))])
    (check-scored ids (fit-predict clusterer (chapter name)) (cluster-columns table) text))
  (check-not-false (member '(0 115 0 118 0 94 1 0 46 87) (cluster-columns raw-table))
                   "the 461 digits' column"))

(test-gpu "embeddings: each digit has a UMAP cluster; 66 of the 91 misplaced are 1s or 9s"
  (define-values (chapter _blocks) (force embeddings))
  (define digit (chapter 'digit))
  (define labels (clusters-of (chapter 'clusterer) (chapter 'embedded) digit))
  (define misplaced (misplaced-by-class labels digit))
  (check-equal? (sort (filter-not negative? (remove-duplicates labels)) <) (range 10))
  (check-equal? misplaced '(0 28 11 2 4 3 1 0 4 38))
  (check-equal? (mismatches labels digit) 91)
  (check-equal? (+ (list-ref misplaced 1) (list-ref misplaced 9)) 66))

(test-gpu "embeddings: the chapter repeats in one process"
  (define-values (chapter blocks) (force embeddings))
  (define-values (_again again) (evaluated-chapter "embeddings.scrbl"))
  (define images (chapter 'images))
  (check-equal? (printed again) (printed blocks))
  (check-equal? (device-matrix->list* (fit-transform (umap) images))
                (device-matrix->list* (chapter 'embedded))
                "UMAP")
  (check-equal? (device-matrix->list* (fit-transform (tsne) images))
                (device-matrix->list* (chapter 'pictured))
                "t-SNE"))

(test-gpu "embeddings: PCA's printed trustworthiness holds over 10 fits"
  (define-values (chapter _blocks) (force embeddings))
  (define images (chapter 'images))
  (define scores
    (for/list ([_ (in-range 10)])
      (trustworthiness images (fit-transform (pca #:n-components 2) images))))
  (check-equal? (remove-duplicates (map rounded scores)) '("0.83"))
  (check <= (- (apply max scores) (apply min scores)) 7.2e-6))
