#lang racket/base

(require (only-in cuml/preprocessing
                  standard-scaler-result-mean
                  standard-scaler-result-scale
                  standard-scaler-result-with-mean
                  standard-scaler-result-with-std)
         (only-in racket/flonum fl* fl+ fl- fl/ flsingle)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in racket/math pi sqr)
         (only-in raft device-vector->list list*->device-matrix list->device-vector)
         (only-in raft/private/array buffer->device-array device-array-buffer)
         (only-in threading ~>))

(provide bits-equal?
         constant-rows
         dyadic-rows
         generated-rows
         host-inverse
         host-transform
         normal-rows
         round-to
         rows->matrix
         standardisation-errors
         strided-vector
         ulp
         unit-roundoff)

(define (rows->matrix rows dtype)
  (list*->device-matrix rows #:dtype dtype))

(define (strided-vector n)
  (define backing
    (list->device-vector (for/list ([i (in-range (* 2 n))])
                           (exact->inexact i))
                         #:dtype 'float64))
  (buffer->device-array (device-array-buffer backing) 0 (list n) '(2) 'float64 'device))

(define (normal-source seed)
  (define g (make-pseudo-random-generator))
  (parameterize ([current-pseudo-random-generator g])
    (random-seed seed))
  (lambda ()
    (parameterize ([current-pseudo-random-generator g])
      (define u (- 1.0 (random)))
      (define v (random))
      (* (sqrt (* -2.0 (log u))) (cos (* 2.0 pi v))))))

(define (column-shape seed d)
  (define g (make-pseudo-random-generator))
  (parameterize ([current-pseudo-random-generator g])
    (random-seed (+ seed 7919))
    (for/list ([_ (in-range d)])
      (cons (- (* 2000.0 (random)) 1000.0) (expt 10.0 (- (* 4.0 (random)) 2.0))))))

(define (normal-rows seed n d dtype)
  (define next (normal-source seed))
  (define shape (column-shape seed d))
  (define narrow (round-to dtype))
  (for/list ([_ (in-range n)])
    (for/list ([column (in-list shape)])
      (match-define (cons centre spread) column)
      (~> (+ centre (* spread (next))) (min 1e4) (max -1e4) narrow))))

(define (generated-rows seed n d dtype value)
  (define next (normal-source seed))
  (define uniform (make-pseudo-random-generator))
  (parameterize ([current-pseudo-random-generator uniform])
    (random-seed (+ seed 104729)))
  (define narrow (round-to dtype))
  (for/list ([_ (in-range n)])
    (for/list ([j (in-range d)])
      (define u
        (parameterize ([current-pseudo-random-generator uniform])
          (random)))
      (narrow (exact->inexact (value (next) u j))))))

(define (dyadic-rows c k)
  (for/list ([i (in-range 8)])
    (define step (expt 2.0 k))
    (list (if (even? i)
              (+ c step)
              (- c step)))))

(define (constant-rows n value)
  (make-list n (list value)))

(define (ulp x)
  (define a (abs x))
  (- (flnext a) a))

(define (flnext x)
  (~> (real->floating-point-bytes x 8)
      (integer-bytes->integer #f)
      add1
      (integer->integer-bytes 8 #f)
      floating-point-bytes->real))

(define (unit-roundoff dtype)
  (if (eq? dtype 'float32)
      (expt 2.0 -24)
      (expt 2.0 -53)))

(define (round-to dtype)
  (if (eq? dtype 'float32) flsingle values))

(define (exact-moments column)
  (define xs (map inexact->exact column))
  (define n (length xs))
  (define mean (/ (apply + xs) n))
  (values mean (~> (for/sum ([x (in-list xs)]) (sqr (- x mean))) (/ n) exact->inexact sqrt)))

(define (standardisation-errors x-column t-column mean scale dtype)
  (define u (unit-roundoff dtype))
  (define narrow (round-to dtype))
  (define-values (x-bar _x-sd) (exact-moments x-column))
  (define-values (t-bar t-sd) (exact-moments t-column))
  (define predicted (/ (- x-bar (inexact->exact (narrow mean))) (inexact->exact (narrow scale))))
  (list (/ (abs (- t-bar predicted)) u)
        (/ (abs t-bar) (* 4 u (+ (/ (abs mean) scale) 1)))
        (/ (abs (- t-sd 1)) (* 4 u))))

(define (bits-equal? xs ys)
  (and (= (length xs) (length ys))
       (for/and ([x (in-list xs)]
                 [y (in-list ys)])
         (cond
           [(list? x) (bits-equal? x y)]
           [else (eqv? x y)]))))

(define (arithmetic dtype)
  (if (eq? dtype 'float32)
      (values (lambda (a b) (flsingle (fl- a b)))
              (lambda (a b) (flsingle (fl/ a b)))
              (lambda (a b) (flsingle (fl* a b)))
              (lambda (a b) (flsingle (fl+ a b)))
              flsingle)
      (values fl- fl/ fl* fl+ values)))

(define (applied-vectors result narrow)
  (define (host v present?)
    (and present? v (list->vector (map narrow (device-vector->list v)))))
  (values (host (standard-scaler-result-mean result) (standard-scaler-result-with-mean result))
          (host (standard-scaler-result-scale result) (standard-scaler-result-with-std result))))

(define (host-transform result rows dtype)
  (define-values (sub div _mul _add narrow) (arithmetic dtype))
  (define-values (means scales) (applied-vectors result narrow))
  (for/list ([row (in-list rows)])
    (for/list ([x (in-list row)]
               [j (in-naturals)])
      (define m (and means (vector-ref means j)))
      (define s (and scales (vector-ref scales j)))
      (cond
        [(and m s) (div (sub x m) s)]
        [m (sub x m)]
        [s (div x s)]
        [else x]))))

(define (host-inverse result rows dtype)
  (define-values (_sub _div mul add narrow) (arithmetic dtype))
  (define-values (means scales) (applied-vectors result narrow))
  (for/list ([row (in-list rows)])
    (for/list ([t (in-list row)]
               [j (in-naturals)])
      (define s (and scales (vector-ref scales j)))
      (define m (and means (vector-ref means j)))
      (cond
        [(and s m) (add (mul t s) m)]
        [s (mul t s)]
        [m (add t m)]
        [else t]))))
