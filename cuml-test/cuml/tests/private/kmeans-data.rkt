#lang racket/base

(require (only-in cuml/cluster/private/blobs blobs)
         (only-in racket/list take)
         (only-in raft
                  device-matrix->list*
                  device-vector->list
                  list*->device-matrix
                  list->device-vector))

(provide argmin-row
         blob-data
         cell-means
         dtypes
         far-start
         first-rows
         floats
         grid64
         grid64-means
         grid64-start
         grid8
         grid8-labels
         grid-weights
         host
         host-vector
         mat
         shift
         squared-distance
         start
         step-weights
         vec
         weighted-grid8
         weighted-start)

(define dtypes '(float32 float64))

(define grid8 '((0 0) (0 2) (2 0) (2 2) (10 10) (10 12) (12 10) (12 12)))
(define grid8-labels '(0 0 0 0 1 1 1 1))
(define start '((1 0) (11 11)))
(define cell-means '((1 1) (11 11)))
(define far-start '((1 1) (11 11) (1000 1000)))
(define weighted-grid8 '((0 0) (0 2) (2 0) (2 2) (10 10) (10 14) (14 10) (14 14)))
(define weighted-start '((1 0) (12 11)))
(define grid-weights '(1 1 1 1 3 3 3 3))
(define step-weights '(1 3 1 3 1 1 1 1))

(define offsets '((0 0) (40 0) (0 40) (40 40)))

(define (shift rows by)
  (for/list ([r (in-list rows)])
    (map + r by)))

(define grid64
  (for*/list ([o (in-list offsets)]
              [a (in-list '(0 2 4 6))]
              [b (in-list '(0 2 4 6))])
    (map + o (list a b))))

(define grid64-start (shift offsets '(1 1)))
(define grid64-means (shift offsets '(3 3)))

(define (floats rows)
  (for/list ([r (in-list rows)])
    (map exact->inexact r)))

(define (mat rows dt)
  (list*->device-matrix rows #:dtype dt))

(define (vec xs dt)
  (list->device-vector xs #:dtype dt))

(define (host m)
  (device-matrix->list* m))

(define (host-vector v)
  (device-vector->list v))

(define (blob-data n d k seed dt #:std [std 0.6])
  (define-values (X _labels) (blobs n d #:centers k #:cluster-std std #:seed seed #:dtype dt))
  X)

(define (first-rows X k dt)
  (mat (take (host X) k) dt))

(define (squared-distance x c)
  (for/sum ([a (in-list x)]
            [b (in-list c)])
    (* (- a b) (- a b))))

(define (argmin-row row)
  (for/fold ([best 0]
             [best-value +inf.0]
             #:result best)
            ([v (in-list row)]
             [j (in-naturals)])
    (if (< v best-value)
        (values j v)
        (values best best-value))))
