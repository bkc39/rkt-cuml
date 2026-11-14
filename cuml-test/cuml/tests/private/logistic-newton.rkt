#lang racket/base

(require (only-in racket/list argmax make-list range take)
         (only-in "exact-algebra.rkt" dot))

(provide newton-coef)

(define newton-iterations 100)
(define newton-step-floor 1e-15)

(define (sigmoid z)
  (cond
    [(>= z 0) (/ 1.0 (+ 1.0 (exp (- z))))]
    [else
     (define e (exp z))
     (/ e (+ 1.0 e))]))

(define (solved A b)
  (define n (length b))
  (define rows
    (for/vector ([row (in-list A)]
                 [v (in-list b)])
      (list->vector (append row (list v)))))
  (define (at r c)
    (vector-ref (vector-ref rows r) c))
  (for ([c (in-range n)])
    (define pivot (argmax (lambda (r) (abs (at r c))) (range c n)))
    (define top (vector-ref rows c))
    (vector-set! rows c (vector-ref rows pivot))
    (vector-set! rows pivot top)
    (for ([r (in-range (add1 c) n)])
      (define f (/ (at r c) (at c c)))
      (for ([j (in-range c (add1 n))])
        (vector-set! (vector-ref rows r) j (- (at r j) (* f (at c j)))))))
  (define x (make-vector n 0.0))
  (for ([r (in-range (sub1 n) -1 -1)])
    (vector-set! x
                 r
                 (/ (- (at r n) (for/sum ([j (in-range (add1 r) n)]) (* (at r j) (vector-ref x j))))
                    (at r r))))
  (vector->list x))

;; The float64 optimum of (1/m) sum_i l_i + lam/2 |coef|^2 for binary labels y in {0, 1},
;; the intercept unpenalised; the coefficients only.
(define (newton-coef X y #:lam lam #:fit-intercept fi)
  (define p (length (car X)))
  (define rows
    (for/list ([r (in-list X)])
      (define xs (map exact->inexact r))
      (if fi
          (append xs '(1.0))
          xs)))
  (define q (length (car rows)))
  (define m (length rows))
  (define (ridge j)
    (if (< j p) lam 0.0))
  (let loop ([theta (make-list q 0.0)]
             [iteration 0])
    (define mu (map (lambda (r) (sigmoid (dot r theta))) rows))
    (define gradient
      (for/list ([j (in-range q)])
        (+ (/ (for/sum ([r (in-list rows)] [u (in-list mu)] [l (in-list y)])
                       (* (- u l) (list-ref r j)))
              m)
           (* (ridge j) (list-ref theta j)))))
    (define hessian
      (for/list ([j (in-range q)])
        (for/list ([k (in-range q)])
          (+ (/ (for/sum ([r (in-list rows)] [u (in-list mu)])
                         (* u (- 1 u) (list-ref r j) (list-ref r k)))
                m)
             (if (= j k)
                 (ridge j)
                 0.0)))))
    (define step (solved hessian gradient))
    (define next (map - theta step))
    (if (or (>= iteration newton-iterations)
            (< (apply max (map abs step)) (* newton-step-floor (apply max 1e-300 (map abs next)))))
        (take next p)
        (loop next (add1 iteration)))))
