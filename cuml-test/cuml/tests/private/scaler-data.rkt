#lang racket/base

(require (only-in cuml/preprocessing
                  standard-scaler-result-mean
                  standard-scaler-result-scale
                  standard-scaler-result-with-mean
                  standard-scaler-result-with-std)
         (only-in racket/flonum fl* fl+ fl- fl/ flsingle)
         (only-in racket/list make-list)
         (only-in racket/math pi)
         (only-in raft device-vector->list list*->device-matrix list->device-vector)
         (only-in raft/private/array buffer->device-array device-array-buffer))

(provide bits-equal?
         constant-rows
         dyadic-rows
         generated-rows
         host-inverse
         host-transform
         normal-rows
         rows->matrix
         strided-vector
         ulp)

(define (rows->matrix rows dtype)
  (list*->device-matrix rows #:dtype dtype))

(define (strided-vector n)
  (define backing (list->device-vector (for/list ([i (in-range (* 2 n))]) (exact->inexact i)) #:dtype 'float64))
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
  (define round-to
    (if (eq? dtype 'float32)
        flsingle
        values))
  (for/list ([_ (in-range n)])
    (for/list ([column (in-list shape)])
      (round-to (max -1e4 (min 1e4 (+ (car column) (* (cdr column) (next)))))))))

(define (generated-rows seed n d dtype value)
  (define next (normal-source seed))
  (define uniform (make-pseudo-random-generator))
  (parameterize ([current-pseudo-random-generator uniform])
    (random-seed (+ seed 104729)))
  (define round-to
    (if (eq? dtype 'float32)
        flsingle
        values))
  (for/list ([_ (in-range n)])
    (for/list ([j (in-range d)])
      (define u
        (parameterize ([current-pseudo-random-generator uniform])
          (random)))
      (round-to (exact->inexact (value (next) u j))))))

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
  (define bits (integer-bytes->integer (real->floating-point-bytes x 8) #f))
  (floating-point-bytes->real (integer->integer-bytes (add1 bits) 8 #f)))

(define (bits-equal? xs ys)
  (and (= (length xs) (length ys))
       (for/and ([x (in-list xs)]
                 [y (in-list ys)])
         (if (list? x)
             (bits-equal? x y)
             (eqv? x y)))))

(define (arithmetic dtype)
  (if (eq? dtype 'float32)
      (values (lambda (a b) (flsingle (fl- a b)))
              (lambda (a b) (flsingle (fl/ a b)))
              (lambda (a b) (flsingle (fl* a b)))
              (lambda (a b) (flsingle (fl+ a b)))
              flsingle)
      (values fl- fl/ fl* fl+ values)))

(define (applied-vectors result round-to)
  (define (host v present?)
    (and present? v (list->vector (map round-to (device-vector->list v)))))
  (values (host (standard-scaler-result-mean result) (standard-scaler-result-with-mean result))
          (host (standard-scaler-result-scale result) (standard-scaler-result-with-std result))))

(define (host-transform result rows dtype)
  (define-values (sub div _mul _add round-to) (arithmetic dtype))
  (define-values (means scales) (applied-vectors result round-to))
  (for/list ([row (in-list rows)])
    (for/list ([x (in-list row)]
               [j (in-naturals)])
      (define centred
        (if means
            (sub x (vector-ref means j))
            x))
      (if scales
          (div centred (vector-ref scales j))
          centred))))

(define (host-inverse result rows dtype)
  (define-values (_sub _div mul add round-to) (arithmetic dtype))
  (define-values (means scales) (applied-vectors result round-to))
  (for/list ([row (in-list rows)])
    (for/list ([t (in-list row)]
               [j (in-naturals)])
      (define scaled
        (if scales
            (mul t (vector-ref scales j))
            t))
      (if means
          (add scaled (vector-ref means j))
          scaled))))
