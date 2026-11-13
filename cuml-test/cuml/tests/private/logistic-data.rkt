#lang racket/base

(require (only-in cuml/linear-model
                  logistic-regression-fit
                  logistic-regression-result-coef
                  logistic-regression-result-intercept
                  logistic-regression-result-n-iter)
         (only-in racket/list make-list)
         (only-in racket/match match)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in "blob-data.rkt" normal-source uniform-source)
         (only-in "linear-data.rkt" by-dtype dyadic on-matrix on-vector))

(provide (struct-out lr-case)
         (struct-out lr-fit)
         (struct-out setting)
         argmax-index
         blob-case
         case-fit
         case-y
         centred-columns
         centred-vector
         cutoff-message
         decision-label
         default-tol-cap
         fit-flat
         fit-parts
         flat-matrix
         gauged
         penalty-settings
         relative-gap
         scaled-case
         scaled-parts
         separable-case
         setting-fit
         tight-max-iter
         tight-tol
         with-column)

(struct lr-case (name X y w) #:transparent)
(struct lr-fit (coef intercept n-iter) #:transparent)
(struct setting (penalty C l1-ratio) #:transparent)

(define penalty-settings
  (list (setting 'l2 1.0 #f) (setting 'l1 1.0 #f) (setting 'elasticnet 0.5 0.5) (setting 'none 1.0 #f)))

(define cutoff-message #rx"first-order residual .* above the bound")

(define tight-max-iter 100000)

(define (tight-tol dt)
  (by-dtype dt 1e-6 1e-10))

(define (default-tol-cap dt)
  (by-dtype dt 8e-2 3e-2))

(define grid-bits 8)
(define weight-levels 4)

(define (blob-case name n p k seed #:sep [sep 1.0] #:scales [scales #f] #:weighted? [weighted? #f])
  (define normal (normal-source seed))
  (define uniform (uniform-source (+ seed 1000)))
  (define centres
    (for/vector ([_ (in-range k)])
      (for/list ([_ (in-range p)])
        (* sep (normal)))))
  (define labels
    (for/list ([_ (in-range n)])
      (min (sub1 k) (inexact->exact (floor (* k (uniform)))))))
  (define rows
    (for/list ([label (in-list labels)])
      (for/list ([c (in-list (vector-ref centres label))]
                 [scale (in-list (or scales (make-list p 1)))])
        (* (dyadic (+ c (normal)) grid-bits) scale))))
  (lr-case name
           rows
           labels
           (and weighted?
                (for/list ([_ (in-range n)])
                  (add1 (inexact->exact (floor (* weight-levels (uniform)))))))))

(define (separable-case name n p seed)
  (define normal (normal-source seed))
  (define rows
    (for/list ([_ (in-range n)])
      (for/list ([_ (in-range p)])
        (dyadic (normal) grid-bits))))
  (define (separated row)
    (match row
      [(cons first rest)
       (define label (if (positive? first) 1 0))
       (values (cons (+ first (if (= label 1) 1/2 -1/2)) rest) label)]))
  (define-values (X y)
    (for/lists (X y) ([row (in-list rows)])
      (separated row)))
  (lr-case name X y #f))

(define (scaled-case c factor)
  (lr-case (lr-case-name c)
           (for/list ([row (in-list (lr-case-X c))])
             (map (lambda (v) (* v factor)) row))
           (lr-case-y c)
           (lr-case-w c)))

(define (scaled-parts parts factor)
  (struct-copy lr-fit
               parts
               [coef
                (for/list ([row (in-list (lr-fit-coef parts))])
                  (map (lambda (v) (* v factor)) row))]))

(define (with-column c value)
  (lr-case (format "~a with a constant column" (lr-case-name c))
           (map (lambda (row) (cons value row)) (lr-case-X c))
           (lr-case-y c)
           (lr-case-w c)))

(define (case-y c [label-dtype 'int32])
  (on-vector (lr-case-y c) label-dtype))

(define (case-fit c
                  dt
                  #:sample-weight [w (on-vector (lr-case-w c) dt)]
                  #:penalty [penalty 'l2]
                  #:C [C 1.0]
                  #:l1-ratio [ratio #f]
                  #:fit-intercept [fi #t]
                  #:max-iter [max-iter 1000]
                  #:tol [tol 1e-4]
                  #:class-weight [class-weight #f])
  (logistic-regression-fit (on-matrix (lr-case-X c) dt)
                           (case-y c)
                           #:sample-weight w
                           #:penalty penalty
                           #:C C
                           #:l1-ratio ratio
                           #:fit-intercept fi
                           #:max-iter max-iter
                           #:tol tol
                           #:class-weight class-weight))

(define (setting-fit c dt s #:fit-intercept [fi #t] #:tol [tol 1e-4] #:max-iter [max-iter 1000])
  (case-fit c
            dt
            #:penalty (setting-penalty s)
            #:C (setting-C s)
            #:l1-ratio (setting-l1-ratio s)
            #:fit-intercept fi
            #:tol tol
            #:max-iter max-iter))

(define (fit-parts r)
  (lr-fit (device-matrix->list* (logistic-regression-result-coef r))
          (device-vector->list (logistic-regression-result-intercept r))
          (logistic-regression-result-n-iter r)))

(define (fit-flat f)
  (append (apply append (lr-fit-coef f)) (lr-fit-intercept f)))

(define (flat-matrix m)
  (apply append (device-matrix->list* m)))

(define (argmax-index xs)
  (for/fold ([best 0]
             [top (car xs)]
             #:result best)
            ([x (in-list (cdr xs))]
             [i (in-naturals 1)])
    (if (> x top)
        (values i x)
        (values best top))))

(define (decision-label z)
  (match z
    [(list s) (if (> s 0) 1 0)]
    [_ (argmax-index z)]))

(define (mean xs)
  (/ (apply + xs) (length xs)))

(define (centred-vector xs)
  (define m (mean xs))
  (map (lambda (v) (- v m)) xs))

(define (centred-columns rows)
  (define means (map mean (apply map list rows)))
  (for/list ([row (in-list rows)])
    (map - row means)))

(define (gauged coef intercept #:centre-coef? [centre-coef? #t])
  (if (= 1 (length coef))
      (append (car coef) intercept)
      (append (apply append
                     (if centre-coef?
                         (centred-columns coef)
                         coef))
              (centred-vector intercept))))

(define smallest-scale 1e-300)

(define (relative-gap ours theirs)
  (define (largest xs)
    (for/fold ([m 0]) ([x (in-list xs)])
      (max m (abs (inexact->exact x)))))
  (/ (largest (map (lambda (o t) (- (inexact->exact o) (inexact->exact t))) ours theirs))
     (max (largest theirs) smallest-scale)))
