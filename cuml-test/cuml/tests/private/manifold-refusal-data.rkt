#lang racket/base

(require (only-in racket/flonum flsingle)
         (only-in racket/match match)
         (only-in "scaler-data.rkt" rows->matrix)
         (only-in "umap-data.rkt" fixture-rows tiny))

(provide cosine-message
         large-message
         large-row
         listed-X
         nan-message
         nan-row
         normless-message
         tiny-with
         zero-row)

(define (tiny-with cells [dt 'float32] [rows (fixture-rows tiny)])
  (rows->matrix (for/list ([row (in-list rows)]
                           [r (in-naturals)])
                  (for/list ([x (in-list row)]
                             [c (in-naturals)])
                    (match (assoc (list r c) cells)
                      [(list _ value) value]
                      [#f x])))
                dt))

(define large (flsingle 1e20))

(define nan-row '([(3 1) +nan.0]))

(define large-row `([(4 0) 0.0] [(4 1) ,large] [(4 2) 0.0]))

(define zero-row '([(3 0) 0.0] [(3 1) 0.0] [(3 2) 0.0]))

(define nan-message "X: row 3, column 1: expected a finite value, got nan")

(define large-message
  (format "X: row 4: expected a squared norm of at most 4.253529586511731e+37, got ~a"
          (* large large)))

(define (normless-message row)
  (format "X: row ~a: expected a squared norm of at least ~a under cosine, got 0"
          row
          (* 3 (expt 2.0 -126))))

(define cosine-message (normless-message 3))

(define listed-X '((0.125 0.0 0.0) (0.5 0.25 0.0)))
