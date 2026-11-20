#lang racket/base

(require (only-in cuml/linear-model
                  logistic-regression-fit
                  logistic-regression-result-coef
                  logistic-regression-result-intercept
                  logistic-regression-result-n-iter)
         (only-in racket/list append* make-list)
         (only-in racket/match match match-define struct*)
         (only-in raft device-matrix->list* device-vector->list device-vector?)
         (only-in "blob-data.rkt" normal-source uniform-source)
         (only-in "common-data.rkt" columns-of mean on-vector scaled-rows)
         (only-in "linear-data.rkt" by-dtype dyadic on-matrix))

(provide (struct-out lr-case)
         (struct-out lr-fit)
         (struct-out setting)
         argmax-index
         blob-case
         case-fit
         case-y
         centred-columns
         centred-vector
         coef-flat
         correct
         cutoff-exn?
         cutoff-message
         decision-label
         default-setting
         default-tol
         default-tol-cap
         fit-flat
         fit-parts
         flat-matrix
         gauged-fit
         gauged-sklearn
         penalty-settings
         relative-gap
         repeated-rows
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

(define default-setting (setting 'l2 1.0 #f))

(define penalty-settings
  (list default-setting (setting 'l1 1.0 #f) (setting 'elasticnet 0.5 0.5) (setting 'none 1.0 #f)))

(define cutoff-message #rx"first-order residual .* above the bound")

(define (cutoff-exn? e)
  (and (exn:fail? e) (regexp-match? cutoff-message (exn-message e))))

(define default-tol 1e-4)

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
    (for/vector #:length k
                ([_ (in-range k)])
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
    (match-define (cons first rest) row)
    (define label (if (positive? first) 1 0))
    (values (cons (+ first (if (= label 1) 1/2 -1/2)) rest) label))
  (for/lists (X y #:result (lr-case name X y #f)) ([row (in-list rows)]) (separated row)))

(define (scaled-case c factor)
  (struct-copy lr-case c [X (scaled-rows (lr-case-X c) factor)]))

(define (scaled-parts parts factor)
  (struct-copy lr-fit parts [coef (scaled-rows (lr-fit-coef parts) factor)]))

(define (with-column c value)
  (lr-case (format "~a with a constant column" (lr-case-name c))
           (map (lambda (row) (cons value row)) (lr-case-X c))
           (lr-case-y c)
           (lr-case-w c)))

(define (repeated-rows c counts)
  (define (spread xs)
    (for*/list ([(x n) (in-parallel (in-list xs) (in-list counts))]
                [_ (in-range n)])
      x))
  (lr-case (lr-case-name c) (spread (lr-case-X c)) (spread (lr-case-y c)) #f))

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
                  #:tol [tol default-tol]
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

(define (setting-fit c
                     dt
                     s
                     #:fit-intercept [fi #t]
                     #:tol [tol default-tol]
                     #:max-iter [max-iter 1000])
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

(define (coef-flat f)
  (append* (lr-fit-coef f)))

(define (fit-flat f)
  (append (coef-flat f) (lr-fit-intercept f)))

(define (flat-matrix m)
  (append* (device-matrix->list* m)))

(define (correct predicted truth)
  (for/sum ([p (in-list predicted)] [t (in-list truth)]) (if (equal? p t) 1 0)))

(define (argmax-index xs)
  (match-define (cons first rest) xs)
  (for/fold ([best 0]
             [top first]
             #:result best)
            ([x (in-list rest)]
             [i (in-naturals 1)])
    (if (> x top)
        (values i x)
        (values best top))))

(define (decision-label z)
  (match z
    [(list s) (if (> s 0) 1 0)]
    [_ (argmax-index z)]))

(define (centred-vector xs)
  (define m (mean xs))
  (map (lambda (v) (- v m)) xs))

(define (centred-columns rows)
  (define means (map mean (columns-of rows)))
  (for/list ([row (in-list rows)])
    (map - row means)))

(define (gauged coef intercept centre-coef?)
  (match coef
    [(list row) (append row intercept)]
    [_
     (append (append* (if centre-coef?
                          (centred-columns coef)
                          coef))
             (centred-vector intercept))]))

(define (gauged-fit f #:centre-coef? [centre-coef? #t])
  (match-define (struct* lr-fit ([coef coef] [intercept intercept])) f)
  (gauged coef intercept centre-coef?))

(define (gauged-sklearn theirs #:centre-coef? [centre-coef? #t])
  (gauged (hash-ref theirs 'coef) (hash-ref theirs 'intercept) centre-coef?))

(define smallest-scale 1e-300)

(define (relative-gap ours theirs)
  (define (largest xs)
    (for/fold ([m 0]) ([x (in-list xs)])
      (max m (abs (inexact->exact x)))))
  (/ (largest (map (lambda (o t) (- (inexact->exact o) (inexact->exact t))) ours theirs))
     (max (largest theirs) smallest-scale)))
