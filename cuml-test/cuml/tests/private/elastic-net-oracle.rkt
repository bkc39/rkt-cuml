#lang racket/base

(require (only-in racket/flonum flvector flvector-ref flvector-set! make-flvector)
         (only-in racket/match match-define struct*)
         (only-in racket/math sgn sqr)
         (only-in "linear-data.rkt"
                  dot
                  eigen-range
                  max-abs
                  min-norm-solve
                  weighted-gram
                  weights-or-ones))

(provide (struct-out en-problem)
         (struct-out en-solution)
         (struct-out kkt)
         alpha-max
         curvatures
         host-kkt
         inexact-norm
         norm2
         oracle-elastic-net
         problem-of
         s2-bound
         s2-rounding)

(struct en-problem (G b mx my total yy) #:transparent)
(struct en-solution (coef intercept mu sigma sigma-y mx) #:transparent)
(struct kkt (residual bound allowance) #:transparent)

(define (problem-of X y w fit-intercept)
  (define-values (G b mx my) (weighted-gram X y w fit-intercept))
  (define weights (weights-or-ones w y))
  (en-problem G
              b
              mx
              my
              (apply + weights)
              (for/sum ([v (in-list y)] [wi (in-list weights)]) (* wi (sqr (- v my))))))

(define (alpha-max p l1-ratio)
  (/ (max-abs (en-problem-b p)) (en-problem-total p) (inexact->exact l1-ratio)))

(define host-cd-stop 1e-15)
(define host-cd-epochs 20000)
(define s2-rounding 64)

(define (diagonal G)
  (for/list ([row (in-list G)]
             [j (in-naturals)])
    (list-ref row j)))

(define (picked xs S)
  (for/list ([x (in-list xs)]
             [i (in-naturals)]
             #:when (memv i S))
    x))

(define (penalties alpha l1-ratio)
  (define a (inexact->exact alpha))
  (define r (inexact->exact l1-ratio))
  (values (* a r) (* a (- 1 r))))

(define (host-cd C c l1 l2)
  (define d (length c))
  (define rows
    (for/vector ([row (in-list C)])
      (apply flvector (map exact->inexact row))))
  (define rhs (apply flvector (map exact->inexact c)))
  (define fl1 (exact->inexact l1))
  (define fl2 (exact->inexact l2))
  (define b (make-flvector d 0.0))
  (let epoch ([k 0])
    (define moved
      (for/fold ([moved 0.0]) ([j (in-range d)])
        (define row (vector-ref rows j))
        (define cjj (flvector-ref row j))
        (cond
          [(and (zero? cjj) (zero? fl2)) moved]
          [else
           (define old (flvector-ref b j))
           (define z
             (- (flvector-ref rhs j)
                (for/sum ([i (in-range d)] #:unless (= i j))
                         (* (flvector-ref row i) (flvector-ref b i)))))
           (define new (/ (* (sgn z) (max 0.0 (- (abs z) fl1))) (+ cjj fl2)))
           (flvector-set! b j new)
           (max moved (abs (- new old)))])))
    (define top
      (for/fold ([m 0.0]) ([j (in-range d)])
        (max m (abs (flvector-ref b j)))))
    (unless (or (<= moved (* host-cd-stop top)) (>= k host-cd-epochs))
      (epoch (add1 k))))
  (for/list ([j (in-range d)])
    (flvector-ref b j)))

(define (sub-matrix C S)
  (for/list ([row (in-list (picked C S))])
    (picked row S)))

(define (exact-on-support C c l1 l2 approx)
  (define S
    (for/list ([a (in-list approx)]
               [j (in-naturals)]
               #:unless (zero? a))
      j))
  (define signs (map (lambda (a) (inexact->exact (sgn a))) (picked approx S)))
  (define bS
    (if (null? S)
        '()
        (min-norm-solve (for/list ([row (in-list (sub-matrix C S))]
                                   [i (in-naturals)])
                          (for/list ([g (in-list row)]
                                     [j (in-naturals)])
                            (if (= i j)
                                (+ g l2)
                                g)))
                        (for/list ([cj (in-list (picked c S))]
                                   [s (in-list signs)])
                          (- cj (* l1 s))))))
  (define on-support
    (for/hasheqv ([j (in-list S)]
                  [b (in-list bS)])
      (values j b)))
  (define coef
    (for/list ([j (in-range (length c))])
      (hash-ref on-support j 0)))
  (define gradient
    (map -
         c
         (for/list ([row (in-list C)])
           (dot row coef))))
  (define signs-hold?
    (or (zero? l1)
        (for/and ([b (in-list bS)]
                  [s (in-list signs)])
          (= (sgn b) s))))
  (define off-support-holds?
    (for/and ([g (in-list gradient)]
              [j (in-naturals)]
              #:unless (memv j S))
      (<= (abs g) l1)))
  (unless (and signs-hold? off-support-holds?)
    (error 'oracle-elastic-net "the certificate failed on support ~a" S))
  coef)

(define (oracle-elastic-net p alpha l1-ratio)
  (match-define (struct* en-problem ([G G] [b b] [mx mx] [my my] [total total] [yy yy])) p)
  (define-values (l1 l2) (penalties alpha l1-ratio))
  (define C
    (for/list ([row (in-list G)])
      (map (lambda (g) (/ g total)) row)))
  (define c (map (lambda (v) (/ v total)) b))
  (define coef (exact-on-support C c l1 l2 (host-cd C c l1 l2)))
  (define-values (_ bottom) (eigen-range G))
  (en-solution coef
               (- my (dot mx coef))
               (+ (max 0.0 (/ bottom (exact->inexact total))) (exact->inexact l2))
               (for/list ([g (in-list (diagonal G))])
                 (sqrt (exact->inexact (/ g total))))
               (sqrt (exact->inexact (/ yy total)))
               mx))

(define coefficient-rounding 2)

(define (curvatures sigma l2)
  (define sigmas (apply + sigma))
  (for/list ([s (in-list sigma)])
    (max (+ (* s s) (exact->inexact l2)) (* s sigmas))))

(define (host-kkt p coef alpha l1-ratio tol n-iter rows roundoff)
  (match-define (struct* en-problem ([G G] [b b] [total total] [yy yy])) p)
  (define-values (l1 l2) (penalties alpha l1-ratio))
  (define c (map inexact->exact coef))
  (define largest (exact->inexact (max-abs c)))
  (define sigma-y (sqrt (exact->inexact (/ yy total))))
  (define kept
    (for/list ([g (in-list (diagonal G))]
               [j (in-naturals)]
               #:unless (zero? g))
      j))
  (define sigma
    (for/list ([g (in-list (picked (diagonal G) kept))])
      (sqrt (exact->inexact (/ g total)))))
  (define-values (worst reach)
    (for/fold ([worst 0.0]
               [reach 0.0])
              ([row (in-list (picked G kept))]
               [bj (in-list (picked b kept))]
               [cj (in-list (picked c kept))]
               [s (in-list sigma)]
               [d (in-list (curvatures sigma l2))])
      (define g (- (/ (- bj (dot row c)) total) (* l2 cj)))
      (define v
        (if (zero? cj)
            (max 0 (- (abs g) l1))
            (abs (- g (* l1 (sgn cj))))))
      (values (max worst (/ (exact->inexact v) d)) (max reach (/ s d)))))
  (define rounding (* roundoff (sqrt n-iter)))
  (define gradient (* (sqrt rows) sigma-y reach))
  (if (zero? largest)
      (kkt worst 0.0 (* rounding gradient))
      (kkt (/ worst largest) tol (* rounding (+ coefficient-rounding (/ gradient largest))))))

(define (s2-bound o coef alpha l1-ratio tol roundoff)
  (match-define (struct* en-solution ([coef c*] [mu mu] [sigma sigma])) o)
  (define-values (_ l2) (penalties alpha l1-ratio))
  (+ (/ (* tol (exact->inexact (max-abs coef)) (norm2 (curvatures sigma l2))) mu)
     (* s2-rounding roundoff (inexact-norm c*))))

(define (norm2 xs)
  (sqrt (for/sum ([x (in-list xs)]) (* x x))))

(define (inexact-norm xs)
  (norm2 (map exact->inexact xs)))
