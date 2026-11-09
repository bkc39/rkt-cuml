#lang racket/base

(require (only-in racket/flonum flvector flvector-ref flvector-set! make-flvector)
         (only-in racket/match match-define)
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
         host-kkt
         oracle-elastic-net
         problem-of)

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

(define (penalties alpha l1-ratio)
  (define a (inexact->exact alpha))
  (define r (inexact->exact l1-ratio))
  (values (* a r) (* a (- 1 r))))

(define (host-cd C c l1 l2)
  (define d (length c))
  (define rows (for/vector ([row (in-list C)]) (apply flvector (map exact->inexact row))))
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
                (for/sum ([i (in-range d)] #:unless (= i j)) (* (flvector-ref row i) (flvector-ref b i)))))
           (define new (/ (* (sgn z) (max 0.0 (- (abs z) fl1))) (+ cjj fl2)))
           (flvector-set! b j new)
           (max moved (abs (- new old)))])))
    (define top (for/fold ([m 0.0]) ([j (in-range d)]) (max m (abs (flvector-ref b j)))))
    (unless (or (<= moved (* 1e-15 top)) (>= k 20000))
      (epoch (add1 k))))
  (for/list ([j (in-range d)])
    (flvector-ref b j)))

(define (sub-matrix C S)
  (for/list ([i (in-list S)])
    (for/list ([j (in-list S)])
      (list-ref (list-ref C i) j))))

(define (exact-on-support C c l1 l2 approx)
  (define d (length c))
  (define S
    (for/list ([j (in-range d)]
               #:unless (zero? (list-ref approx j)))
      j))
  (define signs
    (for/list ([j (in-list S)])
      (sgn (list-ref approx j))))
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
                        (for/list ([j (in-list S)]
                                   [s (in-list signs)])
                          (- (list-ref c j) (* l1 s))))))
  (define coef
    (for/list ([j (in-range d)])
      (define at (index-of S j))
      (if at
          (list-ref bS at)
          0)))
  (define gradient (map - c (for/list ([row (in-list C)]) (dot row coef))))
  (define signs-hold?
    (or (zero? l1)
        (for/and ([b (in-list bS)]
                  [s (in-list signs)])
          (= (sgn b) s))))
  (define off-support-holds?
    (for/and ([j (in-range d)]
              #:unless (index-of S j))
      (<= (abs (list-ref gradient j)) l1)))
  (unless (and signs-hold? off-support-holds?)
    (error 'oracle-elastic-net "the certificate failed on support ~a" S))
  coef)

(define (index-of xs v)
  (for/first ([x (in-list xs)]
              [i (in-naturals)]
              #:when (= x v))
    i))

(define (oracle-elastic-net p alpha l1-ratio)
  (match-define (en-problem G b mx my total yy) p)
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
               (for/list ([row (in-list G)]
                          [j (in-naturals)])
                 (sqrt (exact->inexact (/ (list-ref row j) total))))
               (sqrt (exact->inexact (/ yy total)))
               mx))

(define (host-kkt p coef alpha l1-ratio tol n-iter rows roundoff)
  (match-define (en-problem G b _ _ total yy) p)
  (define-values (l1 l2) (penalties alpha l1-ratio))
  (define c (map inexact->exact coef))
  (define sigma-y (sqrt (exact->inexact (/ yy total))))
  (define-values (worst norms)
    (for/fold ([worst 0.0]
               [norms 0.0])
              ([row (in-list G)]
               [bj (in-list b)]
               [cj (in-list c)]
               [j (in-naturals)]
               #:unless (zero? (list-ref row j)))
      (define g (- (/ (- bj (dot row c)) total) (* l2 cj)))
      (define v
        (if (zero? cj)
            (max 0 (- (abs g) l1))
            (abs (- g (* l1 (sgn cj))))))
      (define sigma (sqrt (exact->inexact (/ (list-ref row j) total))))
      (values (max worst (/ (exact->inexact v) (* sigma sigma-y))) (+ norms sigma))))
  (define spread (/ (* (exact->inexact (max-abs c)) norms) sigma-y))
  (kkt worst (* tol spread) (* roundoff (+ 1 spread) (sqrt (* n-iter rows)))))

(module+ test
  (require (only-in rackunit check-equal? check-true))
  (define p (problem-of '((1 1) (1 2) (2 2) (2 3)) '(6 8 9 11) #f #t))
  (check-equal? (alpha-max p 1) 5/4)
  (check-equal? (alpha-max p 1/2) 5/2)
  (define lasso (oracle-elastic-net p 0.1 1))
  (check-true (< (abs (- (car (en-solution-coef lasso)) 0.6)) 1e-12))
  (check-equal? (en-solution-coef (oracle-elastic-net p 3 1)) '(0 0)))
