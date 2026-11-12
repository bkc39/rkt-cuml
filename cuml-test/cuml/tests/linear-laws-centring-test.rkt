#lang racket/base

(require (only-in racket/flonum flsingle)
         (only-in racket/list split-at)
         (only-in racket/match match-define)
         (only-in rackunit
                  check-equal?
                  check-info-name
                  check-info-value
                  check-pred
                  current-test-case-around
                  exn:test:check-stack
                  exn:test:check?
                  test-case)
         (only-in threading ~>)
         (only-in "private/exact-algebra.rkt" bottom-vector dot min-norm-solve)
         (only-in "private/linear-data.rkt"
                  columns-of
                  lcase
                  lcase-w
                  lcase-X
                  lcase-y
                  oracle-lstsq
                  solution
                  solution-coef
                  unit-roundoff
                  weighted-gram
                  weights-or-ones)
         (only-in "private/linear-laws.rkt" centring-gamma check-lstsq-fit staging staging-of)
         (only-in "private/real-linear-data.rkt" offset-law-arms real-lcase))

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

(define (failed-laws c dt fit exact #:coef-centring? [coef-centring? #t])
  (define failed '())
  (define (record e)
    (define message
      (for/first ([info (in-list (exn:test:check-stack e))]
                  #:when (eq? (check-info-name info) 'message))
        (check-info-value info)))
    (set! failed (cons message failed)))
  (parameterize ([current-test-case-around (lambda (law)
                                             (with-handlers ([exn:test:check? record])
                                               (law)))])
    (check-lstsq-fit c
                     #t
                     (unit-roundoff dt)
                     fit
                     exact
                     (staging-of (lcase-X c) (lcase-y c) (lcase-w c) #t)
                     #:coef-centring? coef-centring?))
  failed)

(define (weighted-mean weights xs)
  (/ (dot weights xs) (apply + weights)))

(define (mean xs)
  (/ (apply + xs) (length xs)))

(define (flonum x)
  (inexact->exact (real->double-flonum x)))

(define (consistent c coef)
  (define weights (weights-or-ones (lcase-w c) (lcase-y c)))
  (define means (map (lambda (column) (weighted-mean weights column)) (columns-of (lcase-X c))))
  (define rounded (map flonum coef))
  (list rounded (flonum (- (weighted-mean weights (lcase-y c)) (dot means rounded)))))

(define (bottom-perturbed-means c u)
  (define X (lcase-X c))
  (define y (lcase-y c))
  (match-define (solution c* _ _) (oracle-lstsq X y #f #t))
  (match-define (staging scale _ _ _ _ _ _) (staging-of X y #f #t))
  (define-values (G _b _mx _my) (weighted-gram X y #f #t))
  (define v
    (~> (for/list ([row (in-list G)]
                   [di (in-list scale)])
          (for/list ([g (in-list row)]
                     [dj (in-list scale)])
            (* di g dj)))
        bottom-vector
        (map inexact->exact _)))
  (define g (centring-gamma c u))
  (define (bound xs)
    (inexact->exact (* g (mean (map abs xs)))))
  (define columns (columns-of X))
  (define size
    (apply min
           (for/list ([column (in-list columns)]
                      [vj (in-list v)]
                      [dj (in-list scale)]
                      #:unless (zero? vj))
             (/ (* dj (bound column)) (abs vj)))))
  (define epsilon
    (for/list ([vj (in-list v)]
               [dj (in-list scale)])
      (/ (* size vj) dj)))
  (define y-epsilon
    (if (positive? (dot epsilon c*))
        (- (bound y))
        (bound y)))
  (values (for/list ([column (in-list columns)]
                     [e (in-list epsilon)])
            (+ (mean column) e))
          (+ (mean y) y-epsilon)))

(test-case "breast-cancer float32 offset by 2^14: the exact solve of the problem centred with medium-kernel means meets T1 to T4"
  (define c (real-lcase 'breast-cancer '((offset . 14) . unweighted) 'float32))
  (define X (lcase-X c))
  (define y (lcase-y c))
  (check-host-fit c
                  (centred-solve X y (map medium-kernel-mean (columns-of X)) (medium-kernel-mean y))))

(test-case "breast-cancer float32 offset by 2^10, rounded c* and serial float32 means: discriminates the r1 T4 term"
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

(for* ([dt (in-list '(float32 float64))]
       [arm (in-list (hash-ref offset-law-arms dt))])
  (test-case (format "breast-cancer ~a ~a: 0.9 c* with a consistent intercept fails a law, c* none"
                     arm
                     dt)
    (define c (real-lcase 'breast-cancer arm dt))
    (define exact (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) #t))
    (define c* (solution-coef exact))
    (check-equal? (failed-laws c dt (consistent c c*) exact) '())
    (check-pred pair? (failed-laws c dt (consistent c (map (lambda (cj) (* 9/10 cj)) c*)) exact))))

(test-case "float32 columns offset by 2^16, means off by their bound along G~'s bottom eigenvector: T1 holds by C1 alone"
  (define rows
    (for/list ([i (in-range 64)])
      (define a (modulo i 8))
      (define b (quotient i 8))
      (list (+ (expt 2 16) a) (+ (expt 2 16) a b) (+ (* 3 a) (* -2 b) (modulo (* 7 i) 5) -2))))
  (define c
    (lcase "offset grid" (map (lambda (row) (list (car row) (cadr row))) rows) (map caddr rows) #f))
  (define X (lcase-X c))
  (define y (lcase-y c))
  (define exact (oracle-lstsq X y #f #t))
  (define-values (means y-mean) (bottom-perturbed-means c (unit-roundoff 'float32)))
  (define fit (centred-solve X y means y-mean))
  (check-equal? (failed-laws c 'float32 fit exact) '())
  (check-pred (lambda (failed) (ormap (lambda (m) (regexp-match? #rx"T1:" m)) failed))
              (failed-laws c 'float32 fit exact #:coef-centring? #f)))
