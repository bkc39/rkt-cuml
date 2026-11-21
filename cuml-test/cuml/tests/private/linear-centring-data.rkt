#lang racket/base

(require (only-in racket/list range split-at)
         (only-in racket/match match-define)
         (only-in racket/string string-prefix?)
         (only-in rackunit
                  check-info-name
                  check-info-value
                  current-test-case-around
                  exn:test:check-stack
                  exn:test:check?)
         (only-in threading ~>)
         (only-in "common-data.rkt" columns-of flonum-exact unit-roundoff)
         (only-in "exact-algebra.rkt" bottom-vector dot float32-of min-norm-solve norm2)
         (only-in "linear-data.rkt"
                  lcase
                  lcase-X
                  lcase-w
                  lcase-y
                  oracle-lstsq
                  solution
                  weighted-gram
                  weights-or-ones)
         (only-in "linear-laws.rkt"
                  centring-gamma
                  check-lstsq-fit
                  device-means
                  shim-means
                  staging
                  staging-of
                  y-centring-gamma))

(provide bottom-grid-fit
         case-staging
         case-weights
         centred-solve
         check-host-fit
         consistent
         device-means
         exact-norm
         failed-laws
         fails?
         mean-radius
         medium-kernel-mean
         moved-solve
         offset-grid
         serial-mean
         sherman-morrison-step
         staged-bottom-vector
         strided-weighted-mean
         weighted-mean
         worst-y-epsilon)

(define (compensated xs)
  (for/fold ([s 0.0]
             [c 0.0])
            ([v (in-list xs)])
    (define t (float32-of (+ s v)))
    (define lost
      (if (>= (abs s) (abs v))
          (float32-of (+ (float32-of (- s t)) v))
          (float32-of (+ (float32-of (- v t)) s))))
    (values t (float32-of (+ c lost)))))

(define (block-sum partials)
  (define raked
    (let rake ([xs partials])
      (cond
        [(null? xs) '()]
        [else
         (define-values (segment rest) (split-at xs 8))
         (cons (for/fold ([p (car segment)]) ([x (in-list (cdr segment))])
                 (float32-of (+ p x)))
               (rake rest))])))
  (let tree ([v (list->vector raked)]
             [offset 1])
    (cond
      [(= offset 32) (vector-ref v 0)]
      [else
       (tree (for/vector ([i (in-range 32)])
               (if (< (+ i offset) 32)
                   (float32-of (+ (vector-ref v i) (vector-ref v (+ i offset))))
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
  (define total (float32-of (+ (block-sum (map car threads)) (block-sum (map cdr threads)))))
  (~> (* total (float32-of (/ 1.0 n))) float32-of inexact->exact))

(define (strided-weighted-mean weights ys)
  (define products (list->vector (map (lambda (w y) (float32-of (* w y))) weights ys)))
  (define n (vector-length products))
  (define blocks (min 8192 (ceiling (/ n 512))))
  (define (block-partial b)
    (define-values (partial-sum partial-c)
      (for/fold ([held 0.0]
                 [held-c 0.0])
                ([ty (in-range 32)])
        (define-values (s c)
          (compensated (for/list ([j (in-range (+ (* 32 b) ty) n (* 32 blocks))])
                         (vector-ref products j))))
        (define t (float32-of (+ held s)))
        (define lost
          (if (>= (abs held) (abs s))
              (float32-of (+ (float32-of (- held t)) s))
              (float32-of (+ (float32-of (- s t)) held))))
        (values t (float32-of (+ held-c (float32-of (+ c lost)))))))
    (float32-of (+ partial-sum partial-c)))
  (define partials (sort (map block-partial (range blocks)) > #:key abs))
  (define total
    (for/fold ([s 0.0]) ([p (in-list partials)])
      (float32-of (+ s p))))
  (~> (/ total (float32-of (apply + weights))) float32-of inexact->exact))

(define (serial-mean xs)
  (define total
    (for/fold ([s 0.0]) ([x (in-list xs)])
      (float32-of (+ s (real->double-flonum x)))))
  (float32-of (/ total (length xs))))

(define (weighted-mean weights xs)
  (/ (dot weights xs) (apply + weights)))

(define (case-weights c)
  (weights-or-ones (lcase-w c) (lcase-y c)))

(define (case-staging c)
  (staging-of (lcase-X c) (lcase-y c) (lcase-w c) #t))

(define (exact-norm xs)
  (~> xs norm2 exact->inexact inexact->exact))

(define (centred-solve c means y-mean)
  (define weights (case-weights c))
  (define columns
    (for/list ([column (in-list (columns-of (lcase-X c)))]
               [m (in-list means)])
      (map (lambda (x) (- x m)) column)))
  (define ys (map (lambda (v) (- v y-mean)) (lcase-y c)))
  (define (weighted-dot a b)
    (for/sum ([wi (in-list weights)] [ai (in-list a)] [bi (in-list b)]) (* wi ai bi)))
  (define coef
    (min-norm-solve (for/list ([a (in-list columns)])
                      (for/list ([b (in-list columns)])
                        (weighted-dot a b)))
                    (for/list ([a (in-list columns)])
                      (weighted-dot a ys))))
  (list coef (- y-mean (dot means coef))))

(define (moved-solve c epsilon y-epsilon)
  (define weights (case-weights c))
  (centred-solve c
                 (for/list ([column (in-list (columns-of (lcase-X c)))]
                            [e (in-list epsilon)])
                   (+ (weighted-mean weights column) e))
                 (+ (weighted-mean weights (lcase-y c)) y-epsilon)))

(define (check-host-fit c fit)
  (check-lstsq-fit c
                   #t
                   (unit-roundoff 'float32)
                   fit
                   (oracle-lstsq (lcase-X c) (lcase-y c) #f #t)
                   (case-staging c)
                   #:means device-means))

(define (failed-laws c
                     dt
                     fit
                     exact
                     #:coef-centring? [coef-centring? #t]
                     #:cap-scale [cap-scale 1]
                     #:means [means shim-means])
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
                     (case-staging c)
                     #:coef-centring? coef-centring?
                     #:cap-scale cap-scale
                     #:means means))
  failed)

(define ((fails? law) failed)
  (for/or ([m (in-list failed)])
    (string-prefix? m (string-append law ":"))))

(define (consistent c coef)
  (define weights (case-weights c))
  (define means (map (lambda (column) (weighted-mean weights column)) (columns-of (lcase-X c))))
  (define rounded (map flonum-exact coef))
  (list rounded (flonum-exact (- (weighted-mean weights (lcase-y c)) (dot means rounded)))))

(define (staged-gram c)
  (match-define (staging scale _ power _ _ _ _) (case-staging c))
  (define-values (G _b _mx _my)
    (weighted-gram (lcase-X c)
                   (lcase-y c)
                   (and (lcase-w c) (map (lambda (v) (* power v)) (lcase-w c)))
                   #t))
  (for/list ([row (in-list G)]
             [di (in-list scale)])
    (for/list ([g (in-list row)]
               [dj (in-list scale)])
      (* di g dj))))

(define (staged-bottom-vector c)
  (~> c staged-gram bottom-vector (map inexact->exact _)))

(define ((gamma-radius c g) xs)
  (inexact->exact (* g (weighted-mean (case-weights c) (map abs xs)))))

(define (mean-radius c u)
  (gamma-radius c (centring-gamma c u)))

(define (worst-y-epsilon c u epsilon)
  (define radius ((gamma-radius c (y-centring-gamma c u)) (lcase-y c)))
  (match-define (solution c* _ _) (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) #t))
  (if (positive? (dot epsilon c*))
      (- radius)
      radius))

(define (bottom-epsilon c u)
  (match-define (staging scale _ _ _ _ _ _) (case-staging c))
  (define v (staged-bottom-vector c))
  (define radius (mean-radius c u))
  (define size
    (apply min
           (for/list ([column (in-list (columns-of (lcase-X c)))]
                      [vj (in-list v)]
                      [dj (in-list scale)]
                      #:unless (zero? vj))
             (/ (* dj (radius column)) (abs vj)))))
  (for/list ([vj (in-list v)]
             [dj (in-list scale)])
    (/ (* size vj) dj)))

(define (sherman-morrison-step c epsilon)
  (match-define (staging scale _ power _ _ _ _) (case-staging c))
  (define staged (map * scale epsilon))
  (* power (apply + (case-weights c)) (dot staged (min-norm-solve (staged-gram c) staged))))

(define (offset-grid power)
  (define rows
    (for/list ([i (in-range 64)])
      (define a (modulo i 8))
      (define b (quotient i 8))
      (list (+ (expt 2 power) a) (+ (expt 2 power) a b) (+ (* 3 a) (* -2 b) (modulo (* 7 i) 5) -2))))
  (lcase (format "offset grid 2^~a" power)
         (for/list ([row (in-list rows)])
           (match-define (list a b _) row)
           (list a b))
         (for/list ([row (in-list rows)])
           (match-define (list _ _ target) row)
           target)
         #f))

(define (bottom-grid-fit c u)
  (define epsilon (bottom-epsilon c u))
  (values epsilon (moved-solve c epsilon (worst-y-epsilon c u epsilon))))
