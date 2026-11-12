#lang racket/base

(require (only-in racket/flonum flsingle)
         (only-in racket/list split-at)
         (only-in racket/match match-define)
         (only-in racket/string string-prefix?)
         (only-in rackunit
                  check-equal?
                  check-info-name
                  check-info-value
                  check-pred
                  check-true
                  current-test-case-around
                  exn:test:check-stack
                  exn:test:check?
                  test-case)
         (only-in threading ~>)
         (only-in "private/exact-algebra.rkt" bottom-vector dot min-norm-solve norm2)
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
                   (case-staging c)))

(define (failed-laws c dt fit exact #:coef-centring? [coef-centring? #t] #:cap-scale [cap-scale 1])
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
                     #:cap-scale cap-scale))
  failed)

(define ((fails? law) failed)
  (for/or ([m (in-list failed)])
    (string-prefix? m (string-append law ":"))))

(define (flonum x)
  (inexact->exact (real->double-flonum x)))

(define (consistent c coef)
  (define weights (case-weights c))
  (define means (map (lambda (column) (weighted-mean weights column)) (columns-of (lcase-X c))))
  (define rounded (map flonum coef))
  (list rounded (flonum (- (weighted-mean weights (lcase-y c)) (dot means rounded)))))

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

(define (mean-radius c u)
  (define weights (case-weights c))
  (define g (centring-gamma c u))
  (lambda (xs) (inexact->exact (* g (weighted-mean weights (map abs xs))))))

(define (worst-y-epsilon c u epsilon)
  (define radius ((mean-radius c u) (lcase-y c)))
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
         (map (lambda (row) (list (car row) (cadr row))) rows)
         (map caddr rows)
         #f))

(define (bottom-grid-fit c u)
  (define epsilon (bottom-epsilon c u))
  (values epsilon (moved-solve c epsilon (worst-y-epsilon c u epsilon))))

(test-case "breast-cancer float32 offset by 2^14: the exact solve of the problem centred with medium-kernel means meets T1 to T4"
  (define c (real-lcase 'breast-cancer '((offset . 14) . unweighted) 'float32))
  (check-host-fit c
                  (centred-solve c
                                 (map medium-kernel-mean (columns-of (lcase-X c)))
                                 (medium-kernel-mean (lcase-y c)))))

(test-case "breast-cancer float32 offset by 2^10, rounded c* and serial float32 means: the residual sum exceeds the r1 T4 term, and T1 to T4 hold by an empirical margin"
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
  (define b (inexact->exact (f32 (- (serial-mean y) products))))
  (define exact-coef (map inexact->exact coef))
  (define weights (case-weights c))
  (define r1-term
    (* (inexact->exact (unit-roundoff 'float32))
       (apply + weights)
       (+ (abs b)
          (for/sum ([column (in-list (columns-of X))] [cj (in-list exact-coef)])
                   (abs (* (weighted-mean weights column) cj))))))
  (define residual-sum
    (for/sum ([row (in-list X)] [yi (in-list y)] [wi (in-list weights)])
             (* wi (- yi b (dot row exact-coef)))))
  (check-true (> (abs residual-sum) r1-term)
              (format "residual sum ~a of the r1 term"
                      (exact->inexact (/ (abs residual-sum) r1-term))))
  (check-host-fit c (list exact-coef b)))

(for* ([dt (in-list '(float32 float64))]
       [arm (in-list (hash-ref offset-law-arms dt))])
  (test-case (format "breast-cancer ~a ~a: c* passes; 0.9 c* fails a law, a halved intercept T4~a"
                     arm
                     dt
                     (if (eq? dt 'float64)
                         ", c* off by 1/10 of its norm along G~'s bottom eigenvector T1"
                         ""))
    (define c (real-lcase 'breast-cancer arm dt))
    (define exact (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) #t))
    (define c* (solution-coef exact))
    (match-define (list rounded b) (consistent c c*))
    (check-equal? (failed-laws c dt (list rounded b) exact) '())
    (check-pred pair? (failed-laws c dt (consistent c (map (lambda (cj) (* 9/10 cj)) c*)) exact))
    (check-pred (fails? "T4") (failed-laws c dt (list rounded (/ b 2)) exact))
    (when (eq? dt 'float64)
      (match-define (staging scale y-scale _ _ _ _ _) (case-staging c))
      (define v (staged-bottom-vector c))
      (define step
        (/ (exact-norm (map (lambda (cj dj) (/ (* y-scale cj) dj)) c* scale)) (* 10 (exact-norm v))))
      (define moved
        (for/list ([cj (in-list c*)]
                   [vj (in-list v)]
                   [dj (in-list scale)])
          (+ cj (/ (* step vj dj) y-scale))))
      (check-pred (fails? "T1") (failed-laws c dt (consistent c moved) exact)))))

(test-case "float32 columns offset by 2^16, means off by their bound along G~'s bottom eigenvector: T1 holds by C1 alone"
  (define c (offset-grid 16))
  (define exact (oracle-lstsq (lcase-X c) (lcase-y c) #f #t))
  (define-values (_epsilon fit) (bottom-grid-fit c (unit-roundoff 'float32)))
  (check-equal? (failed-laws c 'float32 fit exact) '())
  (check-pred (fails? "T1") (failed-laws c 'float32 fit exact #:coef-centring? #f)))

(test-case "float32 columns offset by 2^20, means off by their bound along G~'s bottom eigenvector: C1 is the Sherman–Morrison cap, and T1 fails with it halved"
  (define c (offset-grid 20))
  (define exact (oracle-lstsq (lcase-X c) (lcase-y c) #f #t))
  (define-values (epsilon fit) (bottom-grid-fit c (unit-roundoff 'float32)))
  (define step (sherman-morrison-step c epsilon))
  (check-true (<= 1/2 step 1) (format "the Sherman–Morrison step ~a" (exact->inexact step)))
  (check-equal? (failed-laws c 'float32 fit exact) '())
  (check-pred (fails? "T1") (failed-laws c 'float32 fit exact #:cap-scale 1/2)))

(define corner-signs '(1 -1 -1 0 1 1 0 -1 0 -1 1 1 1 -1 -1 -1 -1 -1 0 -1 -1 1 1 1 -1 1 -1 1 1))

(test-case "breast-cancer float32 offset by 2^10 weighted, means at a point of their error box: T1 holds by C1 and fails without it"
  (define c (real-lcase 'breast-cancer '((offset . 10) . weighted) 'float32))
  (define u (unit-roundoff 'float32))
  (define exact (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) #t))
  (define radius (mean-radius c u))
  (define epsilon
    (for/list ([s (in-list corner-signs)]
               [column (in-list (columns-of (lcase-X c)))])
      (* s (radius column))))
  (define fit (moved-solve c epsilon (worst-y-epsilon c u epsilon)))
  (check-equal? (failed-laws c 'float32 fit exact) '())
  (check-pred (fails? "T1") (failed-laws c 'float32 fit exact #:coef-centring? #f)))
