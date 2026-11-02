#lang racket/base

(require (only-in cuml fit-predict fit-transform pca trustworthiness tsne umap)
         (only-in cuml/scribblings/guide/helpers
                  class-figure
                  cluster-colors
                  cluster-plot
                  clusters-of
                  rounded)
         (only-in pict pict-height pict-width pict?)
         (only-in racket/function curryr)
         (only-in racket/list filter-not last make-list range remove-duplicates)
         (only-in racket/match match-define)
         (only-in racket/promise delay force)
         (only-in racket/string string-contains?)
         (only-in rackunit check check-equal? check-not-equal? check-not-false check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/guide-data.rkt"
                  blended
                  check-scored
                  close-to?
                  cluster-columns
                  drawable-label?
                  embeddings-pin
                  evaluated-chapter
                  flattened-text
                  misplaced-by-class
                  mismatches
                  noise-grey
                  pixels))

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

(define pca-trust-spread 7.2e-6)

(define (printed blocks)
  (map (lambda (forms) (filter-not pict? forms)) blocks))

(define (prose-phrases)
  (match-define (list raw-ari (list raw-largest _ ...) raw-noise) (embeddings-pin 'raw))
  (match-define (list umap-ari _ umap-noise) (embeddings-pin 'umap))
  (match-define (list tsne-ari _ tsne-noise) (embeddings-pin 'tsne))
  (match-define (list pca-trust umap-trust tsne-trust) (embeddings-pin 'trust))
  (match-define (and misplaced (list _ ones _ _ _ _ _ _ _ nines)) (embeddings-pin 'umap-misplaced))
  (define min-cluster-size (embeddings-pin 'min-cluster-size))
  (define min-samples (embeddings-pin 'min-samples))
  (list
   (format "holds ~a handwritten digits" (apply + (embeddings-pin 'digit-sizes)))
   (format "(hdbscan #:min-cluster-size ~a #:min-samples ~a)" min-cluster-size min-samples)
   (format "A cluster needs ~a digits" min-cluster-size)
   (format "a core sample ~a neighbours" min-samples)
   (format "In ~a dimensions" (embeddings-pin 'pixels))
   (format "one of ~a digits mixing" raw-largest)
   (format "and ~a digits noise." raw-noise)
   (format "scoring ~a against ~a on the pixels, and ~a digits noise" umap-ari raw-ari umap-noise)
   (format "Of the ~a digits the clusters misplace, ~a are 1s or 9s"
           (apply + misplaced)
           (+ ones nines))
   (format "scores ~a and calls ~a digits noise" tsne-ari tsne-noise)
   (format "~a against ~a" tsne-trust umap-trust)
   (format "principal components, ~a." pca-trust)))

(test-case "embeddings: the prose states the pinned values"
  (define text
    (flattened-text
     (collection-file-path "embeddings.scrbl" "cuml" "scribblings" "guide" "examples")))
  (for ([phrase (in-list (prose-phrases))])
    (check-true (string-contains? text phrase) phrase)))

(test-gpu "embeddings: the chapter prints its sizes, scores and trustworthiness"
  (define-values (_chapter blocks) (force embeddings))
  (match-define (list data raw umap-block _ tsne-block trust _) blocks)
  (check-equal? (last data) (list (embeddings-pin 'pixels) (embeddings-pin 'digit-sizes)))
  (check-equal? (last raw) (embeddings-pin 'raw))
  (check-equal? (last umap-block) (embeddings-pin 'umap))
  (check-equal? (last tsne-block) (embeddings-pin 'tsne))
  (check-equal? (last trust) (embeddings-pin 'trust)))

(test-gpu "embeddings: the clusters' tables on the pixels, UMAP and t-SNE"
  (define-values (chapter _blocks) (force embeddings))
  (define ids (chapter 'digit-ids))
  (define clusterer (chapter 'clusterer))
  (for ([row (in-list (list (list 'images raw-table 'raw)
                            (list 'embedded umap-table 'umap)
                            (list 'pictured tsne-table 'tsne)))])
    (match-define (list name table pin) row)
    (check-scored ids
                  (fit-predict clusterer (chapter name))
                  (cluster-columns table)
                  (car (embeddings-pin pin))))
  (check-not-false (member '(0 115 0 118 0 94 1 0 46 87) (cluster-columns raw-table))
                   "the 461 digits' column"))

(test-gpu "embeddings: each digit has a UMAP cluster, and the misplaced digits by class"
  (define-values (chapter _blocks) (force embeddings))
  (define labels (clusters-of (chapter 'clusterer) (chapter 'embedded) (chapter 'digit)))
  (check-equal? (sort (filter-not negative? (remove-duplicates labels)) <) (range 10))
  (check-equal? (misplaced-by-class labels (chapter 'digit)) (embeddings-pin 'umap-misplaced)))

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
  (check-equal? (remove-duplicates (map rounded scores)) (list (car (embeddings-pin 'trust))))
  (check <= (- (apply max scores) (apply min scores)) pca-trust-spread))

(define panel (cluster-plot '(0.0) '(0.0) '(0)))

(define panel-targets (append (map blended cluster-colors) (list noise-grey)))

(define all-drawn (make-list (length panel-targets) #t))

(define (panel-colours figure)
  (define drawn (pixels figure #:width (pict-width panel) #:height (pict-height panel)))
  (map (lambda (target) (ormap (close-to? target) drawn)) panel-targets))

(test-gpu "the figures: digits in UMAP's and t-SNE's planes"
  (define-values (chapter blocks) (force embeddings))
  (match-define (list _ _ _ (list umap-figure) _ _ (list tsne-figure)) blocks)
  (define digit (chapter 'digit))
  (define clusterings
    (for/list ([name (in-list '(embedded pictured))])
      (clusters-of (chapter 'clusterer) (chapter name) digit)))
  (for ([labels (in-list clusterings)])
    (check-true (andmap drawable-label? labels)))
  (check-equal? (map (curryr mismatches digit) clusterings)
                (list (apply + (embeddings-pin 'umap-misplaced)) (embeddings-pin 'tsne-mismatches)))
  (for ([figure (in-list (list umap-figure tsne-figure))])
    (check-true (pict? figure))
    (check-equal? (panel-colours figure) all-drawn)))

(test-gpu "the figures' colour check reads the HDBSCAN panel, not the digit panel or the legend"
  (define-values (chapter _blocks) (force embeddings))
  (define digit (chapter 'digit))
  (define embedded (chapter 'embedded))
  (check-not-equal? (panel-colours (class-figure embedded '() digit #:class-title "digit")) all-drawn)
  (check-not-equal? (panel-colours (class-figure embedded
                                                 (list (list "HDBSCAN" (make-list (length digit) -1)))
                                                 digit
                                                 #:class-title "digit"))
                    all-drawn))
