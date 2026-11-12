#lang racket/base

(require (only-in racket/flonum flsingle)
         (only-in racket/list split-at)
         (only-in rackunit test-case)
         (only-in threading ~>)
         (only-in "private/exact-algebra.rkt" dot min-norm-solve)
         (only-in "private/linear-data.rkt"
                  columns-of
                  lcase-X
                  lcase-y
                  oracle-lstsq
                  solution-coef
                  unit-roundoff)
         (only-in "private/linear-laws.rkt" check-lstsq-fit staging-of)
         (only-in "private/real-linear-data.rkt" real-lcase))

(define (f32 x)
  (flsingle (real->double-flonum x)))

(define (compensated xs)
  (for/fold ([s 0.0]
             [c 0.0])
            ([v (in-list xs)])
    (define t (f32 (+ s v)))
    (define lost
      (if (>= (abs s) (abs v))
          (f32 (+ (f32 (- s t)) v))
          (f32 (+ (f32 (- v t)) s))))
    (values t (f32 (+ c lost)))))

(define (block-sum partials)
  (define raked
    (let rake ([xs partials])
      (cond
        [(null? xs) '()]
        [else
         (define-values (segment rest) (split-at xs 8))
         (cons (for/fold ([p (car segment)]) ([x (in-list (cdr segment))])
                 (f32 (+ p x)))
               (rake rest))])))
  (let tree ([v (list->vector raked)]
             [offset 1])
    (cond
      [(= offset 32) (vector-ref v 0)]
      [else
       (tree (for/vector ([i (in-range 32)])
               (if (< (+ i offset) 32)
                   (f32 (+ (vector-ref v i) (vector-ref v (+ i offset))))
                   (vector-ref v i)))
             (* 2 offset))])))

(define (medium-kernel-mean xs)
  (define v (list->vector (map real->double-flonum xs)))
  (define n (vector-length v))
  (define threads
    (for/list ([t (in-range 256)])
      (call-with-values (lambda ()
                          (compensated (for/list ([i (in-range t n 256)])
                                         (vector-ref v i))))
                        cons)))
  (define total (f32 (+ (block-sum (map car threads)) (block-sum (map cdr threads)))))
  (~> (* total (f32 (/ 1.0 n))) f32 inexact->exact))

(define (serial-mean xs)
  (define total
    (for/fold ([s 0.0]) ([x (in-list xs)])
      (f32 (+ s (real->double-flonum x)))))
  (f32 (/ total (length xs))))

(define (centred-solve X y means y-mean)
  (define columns
    (for/list ([column (in-list (columns-of X))]
               [m (in-list means)])
      (map (lambda (x) (- x m)) column)))
  (define ys (map (lambda (v) (- v y-mean)) y))
  (define coef
    (min-norm-solve (for/list ([a (in-list columns)])
                      (for/list ([b (in-list columns)])
                        (dot a b)))
                    (for/list ([a (in-list columns)])
                      (dot a ys))))
  (list coef (- y-mean (dot means coef))))

(define (check-host-fit c fit)
  (check-lstsq-fit c
                   #t
                   (unit-roundoff 'float32)
                   fit
                   (oracle-lstsq (lcase-X c) (lcase-y c) #f #t)
                   (staging-of (lcase-X c) (lcase-y c) #f #t)))

(test-case "breast-cancer float32 offset by 2^14: the exact solve of the problem centred with medium-kernel means meets T1 to T4"
  (define c (real-lcase 'breast-cancer '((offset . 14) . unweighted) 'float32))
  (define X (lcase-X c))
  (define y (lcase-y c))
  (check-host-fit c
                  (centred-solve X y (map medium-kernel-mean (columns-of X)) (medium-kernel-mean y))))

(test-case "breast-cancer float32 offset by 2^10: rounded exact coefficients with serial float32 means meet T1 to T4"
  (define c (real-lcase 'breast-cancer '((offset . 10) . unweighted) 'float32))
  (define X (lcase-X c))
  (define y (lcase-y c))
  (define coef
    (for/list ([cj (in-list (solution-coef (oracle-lstsq X y #f #t)))])
      (f32 cj)))
  (define means (map serial-mean (columns-of X)))
  (define products
    (for/fold ([s 0.0])
              ([m (in-list means)]
               [cj (in-list coef)])
      (f32 (+ s (f32 (* m cj))))))
  (check-host-fit c (list coef (f32 (- (serial-mean y) products)))))
