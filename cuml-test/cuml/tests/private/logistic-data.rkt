#lang racket/base

(require (only-in cuml/linear-model
                  logistic-regression-fit
                  logistic-regression-result-coef
                  logistic-regression-result-intercept
                  logistic-regression-result-n-iter)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in "blob-data.rkt" normal-source uniform-source)
         (only-in "linear-data.rkt" on-matrix on-vector))

(provide (struct-out lr-case)
         argmax-index
         blob-case
         case-fit
         case-y
         centred
         fit-parts
         relative-gap
         separable-case
         with-column)

(struct lr-case (name X y w) #:transparent)

(define (dyadic v)
  (/ (round (* 256 (inexact->exact v))) 256))

(define (blob-case name n p k seed #:sep [sep 1.0] #:scales [scales #f] #:weighted? [weighted? #f])
  (define normal (normal-source seed))
  (define uniform (uniform-source (+ seed 1000)))
  (define centres
    (for/list ([_ (in-range k)])
      (for/list ([_ (in-range p)])
        (* sep (normal)))))
  (define labels
    (for/list ([_ (in-range n)])
      (min (sub1 k) (inexact->exact (floor (* k (uniform)))))))
  (define rows
    (for/list ([label (in-list labels)])
      (for/list ([c (in-list (list-ref centres label))]
                 [j (in-naturals)])
        (* (dyadic (+ c (normal)))
           (if scales
               (list-ref scales j)
               1)))))
  (lr-case name
           rows
           labels
           (and weighted?
                (for/list ([_ (in-range n)])
                  (add1 (inexact->exact (floor (* 4 (uniform)))))))))

(define (separable-case name n p seed)
  (define normal (normal-source seed))
  (define rows
    (for/list ([_ (in-range n)])
      (for/list ([_ (in-range p)])
        (dyadic (normal)))))
  (define labels (map (lambda (row) (if (positive? (car row)) 1 0)) rows))
  (lr-case name
           (for/list ([row (in-list rows)]
                      [label (in-list labels)])
             (cons (+ (car row) (if (= label 1) 1/2 -1/2)) (cdr row)))
           labels
           #f))

(define (with-column c value)
  (lr-case (format "~a with a constant column" (lr-case-name c))
           (map (lambda (row) (cons value row)) (lr-case-X c))
           (lr-case-y c)
           (lr-case-w c)))

(define (case-y c [label-dtype 'int32])
  (on-vector (lr-case-y c) label-dtype))

(define case-fit
  (make-keyword-procedure (lambda (kws vals c dt)
                            (define given (map cons kws vals))
                            (define named
                              (if (or (assq '#:sample-weight given) (not (lr-case-w c)))
                                  given
                                  (cons (cons '#:sample-weight (on-vector (lr-case-w c) dt)) given)))
                            (define sorted (sort named keyword<? #:key car))
                            (keyword-apply logistic-regression-fit
                                           (map car sorted)
                                           (map cdr sorted)
                                           (list (on-matrix (lr-case-X c) dt) (case-y c))))))

(define (fit-parts r)
  (list (device-matrix->list* (logistic-regression-result-coef r))
        (device-vector->list (logistic-regression-result-intercept r))
        (logistic-regression-result-n-iter r)))

(define (argmax-index xs)
  (for/fold ([best 0])
            ([x (in-list xs)]
             [i (in-naturals)]
             #:when (> x (list-ref xs best)))
    i))

(define (centred xs)
  (define (centre row mean)
    (map (lambda (v) (- v mean)) row))
  (cond
    [(real? (car xs)) (centre xs (/ (apply + xs) (length xs)))]
    [else
     (define means
       (for/list ([column (in-list (apply map list xs))])
         (/ (apply + column) (length column))))
     (for/list ([row (in-list xs)])
       (map - row means))]))

(define (relative-gap ours theirs)
  (define (largest xs)
    (for/fold ([m 0]) ([x (in-list xs)])
      (max m (abs (inexact->exact x)))))
  (/ (largest (map (lambda (o t) (- (inexact->exact o) (inexact->exact t))) ours theirs))
     (max (largest theirs) 1e-300)))
