#lang racket/base

(require (only-in cuml/linear-model
                  mbsgd-classifier-result-coef
                  mbsgd-classifier-result-intercept
                  mbsgd-regressor-result-coef
                  mbsgd-regressor-result-intercept)
         (only-in racket/list make-list)
         (only-in raft device-vector->list)
         (only-in "blob-data.rkt" normal-source))

(provide blob-rows
         clf-parts
         dtype-penalties
         learning-rates
         normal-rows
         penalties
         reg-parts
         reg-target
         scaled
         tiled-rows
         with-ones)

(define penalties '(none l1 l2 elasticnet))
(define learning-rates '(constant invscaling adaptive))
(define dtype-penalties '(none l1 l2))

(define (normal-rows n d seed)
  (define next (normal-source seed))
  (for/list ([_ (in-range n)])
    (for/list ([_ (in-range d)])
      (next))))

(define (tiled-rows n d seed #:distinct [distinct 4096])
  (define base (list->vector (normal-rows distinct d seed)))
  (for/list ([i (in-range n)])
    (vector-ref base (modulo i distinct))))

(define (reg-target rows seed #:offset [offset 3.0] #:noise [noise 0.1])
  (define next (normal-source (+ 7919 seed)))
  (define beta
    (for/list ([_ (in-list (car rows))])
      (next)))
  (for/list ([r (in-list rows)])
    (+ offset
       (for/sum ([x (in-list r)]
                 [b (in-list beta)])
         (* x b))
       (* noise (next)))))

(define (blob-rows n d seed #:sep [sep 2.0] #:minority [minority 0.5])
  (define next (normal-source seed))
  (define mu
    (for/list ([_ (in-range d)])
      (* sep (next))))
  (define labels
    (for/list ([i (in-range n)])
      (if (< (/ (+ i 0.5) n) minority) 1 0)))
  (values (for/list ([c (in-list labels)])
            (for/list ([m (in-list mu)])
              (+ (next)
                 (if (= c 1)
                     (/ m 2)
                     (/ m -2)))))
          labels))

(define (scaled rows factor)
  (for/list ([r (in-list rows)])
    (map (lambda (x) (* x factor)) r)))

(define (with-ones rows)
  (for/list ([r (in-list rows)])
    (append r (make-list 1 1.0))))

(define (reg-parts r)
  (list (device-vector->list (mbsgd-regressor-result-coef r)) (mbsgd-regressor-result-intercept r)))

(define (clf-parts r)
  (list (device-vector->list (mbsgd-classifier-result-coef r))
        (mbsgd-classifier-result-intercept r)))
