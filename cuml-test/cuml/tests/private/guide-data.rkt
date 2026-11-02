#lang racket/base

(require (only-in cuml adjusted-rand-score standard-scaler-fit-transform table->device-matrix)
         (only-in cuml/scribblings/guide/helpers cluster-colors counts-of renumbered rounded)
         (only-in cuml/scribblings/utils make-cuml-eval)
         (only-in data/order datum-order order-<?)
         (only-in datasets/core load-dataset-table)
         (only-in pict pict->bitmap)
         (only-in racket/class send)
         (only-in racket/file file->string)
         (only-in racket/function curry)
         (only-in racket/list append-map count remove-duplicates take)
         (only-in racket/match match)
         (only-in racket/sandbox sandbox-namespace-specs)
         (only-in rackunit check-= check-equal?)
         (only-in raft device-vector->list)
         (only-in syntax/modread with-module-reading-parameterization)
         (only-in threading ~>)
         (only-in "scaler-data.rkt" transpose))

(provide blended
         boundary-margin
         check-scored
         close-to?
         cluster-columns
         drawable-label?
         drawn?
         embeddings-pin
         evaluated-chapter
         example-blocks
         flattened-text
         host
         misplaced-by-class
         mismatches
         noise-grey
         painted
         pixels
         read-document
         scaled-matrix
         table-ari)

(define embeddings-pins
  (hasheq 'pixels 64
          'digit-sizes '(183 182 182 181 181 180 179 178 177 174)
          'min-cluster-size 60
          'min-samples 10
          'raw '("0.284" (461 174 160 134 133 102) 633)
          'umap '("0.899" (208 196 185 181 178 178 178 177 166 144) 6)
          'tsne '("0.745" (216 193 180 177 173 172 162 151 114 72) 187)
          'trust '("0.83" "0.988" "0.992")
          'umap-misplaced '(0 28 11 2 4 3 1 0 4 38)
          'tsne-mismatches 330))

(define (embeddings-pin key)
  (hash-ref embeddings-pins key))

(define (read-document file)
  (call-with-input-file file
                        (lambda (in) (with-module-reading-parameterization (lambda () (read in))))))

(define (flattened-text path)
  (regexp-replace* #px"\\s+" (file->string path) " "))

(define (block-forms arguments)
  (match arguments
    [(list* (? keyword?) _ more) (block-forms more)]
    [forms forms]))

(define (example-blocks datum)
  (match datum
    [(list* 'examples arguments) (list (block-forms arguments))]
    [(? list?) (append-map example-blocks datum)]
    [_ '()]))

(define (evaluated-chapter name #:blocks [n #f])
  (define chapter
    (parameterize ([sandbox-namespace-specs (append (sandbox-namespace-specs) '(cuml pict raft))])
      (make-cuml-eval)))
  (define blocks
    (~> (collection-file-path name "cuml" "scribblings" "guide" "examples")
        read-document
        example-blocks))
  (values chapter
          (map (curry map chapter)
               (if n
                   (take blocks n)
                   blocks))))

(define (misplaced-by-class labels classes)
  (define truth (renumbered classes))
  (map (lambda (class)
         (count (lambda (cluster id) (and (= id class) (not (= cluster id)))) labels truth))
       (remove-duplicates truth)))

(define (mismatches labels classes)
  (apply + (misplaced-by-class labels classes)))

(define (drawable-label? label)
  (and (exact-integer? label) (<= -1 label (sub1 (length cluster-colors)))))

(define noise-grey '(137 135 129))

(define channel-tolerance 2)

(define (pixels picture #:width [width #f] #:height [height #f])
  (define bitmap (pict->bitmap picture))
  (define w
    (if width
        (min width (send bitmap get-width))
        (send bitmap get-width)))
  (define h
    (if height
        (min height (send bitmap get-height))
        (send bitmap get-height)))
  (define argb (make-bytes (* 4 w h)))
  (send bitmap get-argb-pixels 0 0 w h argb)
  (for/list ([offset (in-range 0 (bytes-length argb) 4)])
    (list (bytes-ref argb (+ offset 1)) (bytes-ref argb (+ offset 2)) (bytes-ref argb (+ offset 3)))))

(define (blended color)
  (map (lambda (channel) (+ (* 4/5 channel) (* 1/5 255))) color))

(define (close-to? target)
  (lambda (pixel)
    (for/and ([channel (in-list pixel)]
              [expected (in-list target)])
      (<= (abs (- channel expected)) channel-tolerance))))

(define (painted picture target)
  (count (close-to? target) (pixels picture)))

(define (drawn? picture target)
  (positive? (painted picture target)))

(define (scaled-matrix name columns)
  (define-values (_fitted scaled)
    (~> (load-dataset-table name)
        (table->device-matrix #:columns columns)
        standard-scaler-fit-transform))
  scaled)

(define (pairs n)
  (/ (* n (sub1 n)) 2))

(define (pair-sum counts)
  (apply + (map pairs counts)))

(define (sums rows)
  (map (lambda (row) (apply + row)) rows))

(define (table-ari rows)
  (define within (pair-sum (apply append rows)))
  (define row-pairs (pair-sum (sums rows)))
  (define column-pairs (pair-sum (sums (transpose rows))))
  (define expected (/ (* row-pairs column-pairs) (pairs (apply + (sums rows)))))
  (define most (/ (+ row-pairs column-pairs) 2))
  (cond
    [(= most expected) 1]
    [else (/ (- within expected) (- most expected))]))

(define (cluster-columns counts)
  (sort (transpose counts) (order-<? datum-order)))

(define (host v n)
  (take (device-vector->list v) n))

(define (boundary-margin x places)
  (define scale (expt 10 places))
  (define scaled (* (inexact->exact x) scale))
  (/ (abs (- (- scaled (floor scaled)) 1/2)) scale))

(define (check-scored ids labels columns text)
  (check-equal? (cluster-columns (counts-of ids labels)) columns "the table's columns")
  (define ari (adjusted-rand-score ids labels))
  (check-= ari (table-ari columns) 1e-12 "the ARI is the table's")
  (check-equal? (rounded (table-ari columns)) text "the table's ARI, rounded")
  (check-equal? (rounded ari) text "the ARI, rounded"))
