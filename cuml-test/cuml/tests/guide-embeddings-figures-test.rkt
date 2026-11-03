#lang racket/base

(require (only-in cuml/scribblings/guide/helpers class-figure cluster-colors cluster-plot clusters-of)
         (only-in pict pict-height pict-width pict?)
         (only-in racket/function curry curryr)
         (only-in racket/list count last make-list range)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-false check-not-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/guide-data.rkt"
                  blended
                  close-to?
                  drawable-label?
                  embeddings-pin
                  evaluated-chapter
                  mismatches
                  noise-shade?
                  pixels))

(define panel (cluster-plot '(0.0) '(0.0) '(0)))

(define all-drawn (make-list (add1 (length cluster-colors)) #t))

(define (panel-pixels figure)
  (pixels figure #:width (pict-width panel) #:height (pict-height panel)))

(define (greys drawn)
  (count noise-shade? drawn))

(define (panel-colours figure baseline)
  (define drawn (panel-pixels figure))
  (append (map (lambda (colour) (ormap (close-to? colour) drawn)) (map blended cluster-colors))
          (list (> (greys drawn) (greys (panel-pixels baseline))))))

(define (noiseless labels)
  (map (curry max 0) labels))

(define ((hdbscan-figure . samples) labels classes)
  (apply class-figure
         #:class-title "digit"
         (append samples (list (list (list "HDBSCAN" labels)) classes))))

(test-gpu "embeddings: the figures, digits in UMAP's and t-SNE's planes"
  (define-values (chapter blocks) (evaluated-chapter "embeddings.scrbl"))
  (match-define (list _ _ _ (list umap-figure) _ _ (list tsne-figure)) blocks)
  (define digit (chapter 'digit))
  (define names '(embedded pictured))
  (define clusterings
    (for/list ([name (in-list names)])
      (clusters-of (chapter 'clusterer) (chapter name) digit)))
  (for ([labels (in-list clusterings)])
    (check-true (andmap drawable-label? labels)))
  (check-equal? (map (curryr mismatches digit) clusterings)
                (list (apply + (embeddings-pin 'umap-misplaced)) (embeddings-pin 'tsne-mismatches)))
  (for ([figure (in-list (list umap-figure tsne-figure))]
        [name (in-list names)]
        [labels (in-list clusterings)])
    (check-true (pict? figure))
    (define baseline ((hdbscan-figure (chapter name)) (noiseless labels) digit))
    (check-equal? (panel-colours figure baseline) all-drawn (symbol->string name))))

(define grid-size 400)

(define grid
  (hdbscan-figure (map (lambda (i) (exact->inexact (modulo i 20))) (range grid-size))
                  (map (lambda (i) (exact->inexact (quotient i 20))) (range grid-size))))

(define grid-digits (map (curryr modulo 10) (range grid-size)))

(define (with-noise labels)
  (for/list ([label (in-list labels)]
             [i (in-naturals)])
    (if (zero? (modulo i 37)) -1 label)))

(test-case "embeddings: the colour check finds every colour and noise in a panel that draws them"
  (define labels (with-noise grid-digits))
  (check-equal? (panel-colours (grid labels grid-digits) (grid (noiseless labels) grid-digits))
                all-drawn))

(test-case "embeddings: the colour check finds no noise in a panel that draws it in a cluster's colour"
  (define labels (with-noise grid-digits))
  (check-false (last (panel-colours (grid (map (lambda (label) (if (negative? label) 9 label)) labels)
                                          grid-digits)
                                    (grid (noiseless labels) grid-digits)))))

(test-case "embeddings: the colour check reads the HDBSCAN panel, not the digit panel or the legend"
  (define half (with-noise (map (curryr modulo 5) (range grid-size))))
  (check-not-equal? (panel-colours (grid half grid-digits) (grid (noiseless half) grid-digits))
                    all-drawn)
  (define unclustered (grid (make-list grid-size -1) grid-digits))
  (check-not-equal? (panel-colours unclustered (grid (make-list grid-size 0) grid-digits)) all-drawn))
