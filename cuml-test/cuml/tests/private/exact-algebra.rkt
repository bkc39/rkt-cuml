#lang racket/base

(require (only-in racket/list argmax last splitf-at)
         (only-in racket/match match match-define)
         (only-in threading ~>))

(provide bottom-vector
         bracket-condition
         condition
         dot
         eigen-bottom
         eigen-bracket
         eigen-range
         integer-scaled
         mat*vec
         min-norm-solve
         norm2
         outer-bracket
         varying-inner)

(define (dot u v)
  (for/sum ([a (in-list u)] [b (in-list v)]) (* a b)))

(define (norm2 xs)
  (sqrt (dot xs xs)))

(define (common-denominator xs)
  (for/fold ([d 1]) ([x (in-list xs)])
    (lcm d (denominator x))))

(define (integer-scaled xs)
  (define d (common-denominator xs))
  (cons d (map (lambda (x) (* d x)) xs)))

(define (mat*vec A v)
  (cond
    [(and (andmap exact? v) (andmap (lambda (row) (andmap exact? row)) A))
     (define dA (common-denominator (apply append A)))
     (match-define (cons dv iv) (integer-scaled v))
     (for/list ([row (in-list A)])
       (/ (dot (map (lambda (x) (* dA x)) row) iv) (* dA dv)))]
    [else
     (for/list ([row (in-list A)])
       (dot row v))]))

(define (swapped M r p)
  (define row-r (list-ref M r))
  (define row-p (list-ref M p))
  (for/list ([row (in-list M)]
             [i (in-naturals)])
    (cond
      [(= i r) row-p]
      [(= i p) row-r]
      [else row])))

(define (pivoted M r c)
  (define row (list-ref M r))
  (define pivot (map (lambda (x) (/ x (list-ref row c))) row))
  (for/list ([row (in-list M)]
             [i (in-naturals)])
    (define f (list-ref row c))
    (cond
      [(= i r) pivot]
      [(zero? f) row]
      [else (map (lambda (x p) (- x (* f p))) row pivot)])))

(define (pivot-row M r c)
  (for/first ([row (in-list (list-tail M r))]
              [i (in-naturals r)]
              #:unless (zero? (list-ref row c)))
    i))

(define (augmented A b)
  (map (lambda (row bi) (append row (list bi))) A b))

(define (rref A b)
  (define d (length A))
  (let loop ([M (augmented A b)]
             [r 0]
             [c 0]
             [pivots '()])
    (cond
      [(or (= r d) (= c d)) (values M (reverse pivots))]
      [(pivot-row M r c)
       =>
       (lambda (p) (loop (pivoted (swapped M r p) r c) (add1 r) (add1 c) (cons c pivots)))]
      [else (loop M r (add1 c) pivots)])))

(define (scaled-to-integers rows)
  (define d (common-denominator (apply append rows)))
  (for/list ([row (in-list rows)])
    (map (lambda (x) (* d x)) row)))

(define (pivot-first M)
  (define-values (zeros rest) (splitf-at M (lambda (row) (zero? (car row)))))
  (and (pair? rest) (cons (car rest) (append zeros (cdr rest)))))

(define (eliminated rows pivot pivot-tail previous)
  (for/list ([row (in-list rows)])
    (match-define (cons f tail) row)
    (map (lambda (x p) (quotient (- (* pivot x) (* f p)) previous)) tail pivot-tail)))

(define (bareiss M previous)
  (define arranged (pivot-first M))
  (cond
    [(null? M) '()]
    [(not arranged) #f]
    [else
     (match-define (cons (and top (cons pivot pivot-tail)) rest) arranged)
     (define below (bareiss (eliminated rest pivot pivot-tail previous) pivot))
     (and below (cons top below))]))

(define (back-substituted rows)
  (cond
    [(null? rows) '()]
    [else
     (match-define (cons determinant _) (last rows))
     (define scaled
       (for/foldr ([x '()])
                  ([row (in-list rows)])
                  (match-define (cons pivot tail) row)
                  (cons (/ (- (* determinant (last tail)) (dot tail x)) pivot) x)))
     (map (lambda (x) (/ x determinant)) scaled)]))

(define (nonsingular-solve A b)
  (define M (augmented A b))
  (define rows
    (and (for*/and ([row (in-list M)]
                    [x (in-list row)])
           (exact? x))
         (bareiss (scaled-to-integers M) 1)))
  (and rows (back-substituted rows)))

(define (min-norm-solve A b)
  (or (nonsingular-solve A b) (rref-min-norm-solve A b)))

(define (rref-min-norm-solve A b)
  (define d (length A))
  (define-values (M pivots) (rref A b))
  (define pivot-rows
    (for/hash ([c (in-list pivots)]
               [row (in-list M)])
      (values c row)))
  (define p
    (for/list ([c (in-range d)])
      (cond
        [(hash-ref pivot-rows c #f)
         =>
         last]
        [else 0])))
  (define N
    (for/list ([f (in-range d)]
               #:unless (hash-has-key? pivot-rows f))
      (for/list ([c (in-range d)])
        (cond
          [(= c f) 1]
          [(hash-ref pivot-rows c #f)
           =>
           (lambda (row) (- (list-ref row f)))]
          [else 0]))))
  (cond
    [(null? N) p]
    [else
     (define alpha
       (min-norm-solve (for/list ([u (in-list N)])
                         (for/list ([v (in-list N)])
                           (dot u v)))
                       (for/list ([u (in-list N)])
                         (dot u p))))
     (for/fold ([x p])
               ([a (in-list alpha)]
                [u (in-list N)])
       (map (lambda (xj uj) (- xj (* a uj))) x u))]))

(define (unit v)
  (define largest
    (for/fold ([m 0.0]) ([x (in-list v)])
      (max m (abs x))))
  (define scaled (map (lambda (x) (/ x largest)) v))
  (define norm (sqrt (dot scaled scaled)))
  (and (positive? largest)
       (< largest +inf.0)
       (andmap rational? scaled)
       (map (lambda (x) (/ x norm)) scaled)))

(define (rayleigh G v)
  (define e (map inexact->exact v))
  (/ (dot e (mat*vec G e)) (dot e e)))

(define (iterated step start steps)
  (for/fold ([v start]) ([_ (in-range steps)])
    (or (unit (step v)) v)))

(define singular-pivot (expt 2.0 -60))

(define (float-eliminated M)
  (cond
    [(null? M) '()]
    [else
     (define top (argmax (lambda (row) (abs (car row))) M))
     (match-define (cons pivot pivot-tail) top)
     (define p (if (zero? pivot) singular-pivot pivot))
     (cons (cons p pivot-tail)
           (float-eliminated (for/list ([row (in-list (remq top M))])
                               (match-define (cons f tail) row)
                               (map (lambda (x q) (- x (* (/ f p) q))) tail pivot-tail))))]))

(define (float-solve A b)
  (back-substituted (float-eliminated (augmented A b))))

(define (inexact-rows G)
  (for/list ([row (in-list G)])
    (map exact->inexact row)))

(define (start-vector G)
  (for/list ([i (in-range (length G))])
    (/ 1.0 (+ i 1))))

(define (bottom-vector G)
  (define G-inexact (inexact-rows G))
  (iterated (lambda (v) (float-solve G-inexact v)) (start-vector G) 3))

(define (rayleigh-range G)
  (define G-inexact (inexact-rows G))
  (define top-vector (iterated (lambda (v) (mat*vec G-inexact v)) (start-vector G) 300))
  (values (rayleigh G top-vector) (rayleigh G (bottom-vector G))))

(define (positive-definite? A)
  (let loop ([M (scaled-to-integers A)]
             [previous 1])
    (match M
      ['() #t]
      [(cons (cons pivot pivot-tail) rest)
       (and (positive? pivot) (loop (eliminated rest pivot pivot-tail previous) pivot))])))

(define (shifted G s)
  (for/list ([row (in-list G)]
             [i (in-naturals)])
    (for/list ([g (in-list row)]
               [j (in-naturals)])
      (if (= i j)
          (- g s)
          g))))

(define (negated G)
  (for/list ([row (in-list G)])
    (map - row)))

(define bracket-margin (expt 2 -10))

(define (binade x)
  (- (integer-length (numerator x)) (integer-length (denominator x))))

(define (coarse x)
  (define step (expt 2 (- (binade x) 14)))
  (~> (/ x step) round (* step _)))

(define (narrowed holds? fails holds)
  (define mid (~> (+ fails holds) (/ 2) coarse))
  (cond
    [(<= (abs (- fails holds)) (* 2 bracket-margin (max fails holds))) (values fails holds)]
    [(holds? mid) (narrowed holds? fails mid)]
    [else (narrowed holds? mid holds)]))

(define (bottom-bracket G bottom)
  (define (holds? s)
    (positive-definite? (shifted G s)))
  (define (widened fails s)
    (if (holds? s)
        (narrowed holds? fails s)
        (widened s (/ s 2))))
  (define first (~> (- 1 bracket-margin) (* bottom _) coarse))
  (cond
    [(holds? first) (narrowed holds? bottom first)]
    [(positive-definite? G) (widened first (/ first 2))]
    [else (values 0 0)]))

(define (top-bracket G top)
  (define (holds? t)
    (positive-definite? (shifted (negated G) (- t))))
  (define (widened fails t)
    (if (holds? t)
        (narrowed holds? fails t)
        (widened t (* 2 t))))
  (if (positive? top)
      (widened top (~> (+ 1 bracket-margin) (* top _) coarse))
      (values 0 (* (length G) (largest-entry G)))))

(define (largest-entry G)
  (for*/fold ([m 0])
             ([row (in-list G)]
              [g (in-list row)])
    (max m (abs g))))

(define (eigen-bracket G)
  (define largest (largest-entry G))
  (define scale
    (if (zero? largest)
        1
        (expt 2 (binade largest))))
  (define (rescaled pair)
    (match-define (cons top bottom) pair)
    (cons (* scale top) (* scale bottom)))
  (define unit-G
    (for/list ([row (in-list G)])
      (map (lambda (g) (/ g scale)) row)))
  (define-values (top bottom) (rayleigh-range unit-G))
  (define-values (top-in top-out) (top-bracket unit-G top))
  (define-values (bottom-in bottom-out) (bottom-bracket unit-G bottom))
  (values (rescaled (cons top-in bottom-in)) (rescaled (cons top-out bottom-out))))

(define (outer-bracket G)
  (let-values ([(_ outer) (eigen-bracket G)])
    outer))

(define (eigen-range G)
  (match-define (cons top bottom) (outer-bracket G))
  (values top bottom))

(define (eigen-bottom G)
  (cdr (outer-bracket G)))

(define (bracket-condition pair)
  (match-define (cons top bottom) pair)
  (if (> bottom (* 1e-14 top))
      (~> (/ top bottom) exact->inexact sqrt)
      +inf.0))

(define (condition G)
  (bracket-condition (outer-bracket G)))

(define (varying-inner G inner)
  (define kept
    (for/list ([row (in-list G)]
               [i (in-naturals)]
               #:unless (zero? (list-ref row i)))
      i))
  (define (kept-of row)
    (map (lambda (i) (list-ref row i)) kept))
  (cond
    [(= (length kept) (length G)) inner]
    [(null? kept) (cons 0 0)]
    [else
     (define-values (varying _) (eigen-bracket (map kept-of (kept-of G))))
     varying]))

(module+ test
  (require (only-in rackunit check-equal? check-true))
  (define A '((2 1 0) (1 3 1) (0 1 4)))
  (define b '(1 2 3))
  (check-equal? (min-norm-solve A b) (rref-min-norm-solve A b))
  (check-equal? (min-norm-solve '((0 1) (1 0)) '(2 3)) '(3 2))
  (check-equal? (min-norm-solve '() '()) '())
  (check-equal? (mat*vec '((1/3 2) (1/6 -1/4)) '(3/7 1/2)) '(8/7 -3/56))
  (check-equal? (mat*vec '((1.0 2)) '(1/2 1)) '(2.5))
  (check-equal? (min-norm-solve '((0 2 1) (3 0 1/2) (1 1 5)) '(1 2 3))
                (rref-min-norm-solve '((0 2 1) (3 0 1/2) (1 1 5)) '(1 2 3)))
  (check-equal? (min-norm-solve '((1 2) (2 4)) '(1 2)) '(1/5 2/5))
  (check-equal? (nonsingular-solve '((1 2) (2 4)) '(1 2)) #f)
  (check-equal? (nonsingular-solve '((1.0 0) (0 1)) '(1 2)) #f)
  (check-equal? (min-norm-solve '((1/3 1/7) (1/7 1/5)) '(1/2 1/9))
                (rref-min-norm-solve '((1/3 1/7) (1/7 1/5)) '(1/2 1/9)))
  (define-values (top bottom) (rayleigh-range '((4 0 0) (0 2 0) (0 0 1/1000))))
  (check-true (and (exact? top) (exact? bottom)))
  (check-true (<= (* 999/1000 4) top 4))
  (check-true (<= 1/1000 bottom (* 1001/1000 1/1000)))
  (match-define (list b1 b2 b3) (bottom-vector '((4 0 0) (0 2 0) (0 0 1/1000))))
  (check-true (and (< (abs b1) 1e-6) (< (abs b2) 1e-6) (< 0.999999 (abs b3))))
  (define (bracketed? G low high)
    (define-values (hi lo) (eigen-range G))
    (and (<= (* (- 1 (* 2 bracket-margin)) low) lo low)
         (<= high hi (* (+ 1 (* 2 bracket-margin)) high))))
  (define hilbert
    (for/list ([i (in-range 6)])
      (for/list ([j (in-range 6)])
        (/ 1 (+ i j 1)))))
  (check-true (bracketed? hilbert 1.0828e-7 1.6188))
  (check-true (bracketed? (for/list ([row (in-list hilbert)])
                            (map (lambda (h) (* h #e1e400)) row))
                          (* #e1.0828e-7 #e1e400)
                          (* #e1.6188 #e1e400)))
  (define-values (zero-top zero-bottom) (eigen-range '((0 0) (0 0))))
  (check-equal? (list zero-top zero-bottom) '(0 0))
  (define rotation '((1/3 2/3 2/3) (2/3 1/3 -2/3) (2/3 -2/3 1/3)))
  (define spectrum (list 1 (expt 10 -8) (* 2 (expt 10 -8))))
  (define (rotated-by eigenvalues [scale 1])
    (for/list ([u (in-list rotation)])
      (for/list ([v (in-list rotation)])
        (for/sum ([a (in-list u)] [b (in-list v)] [s (in-list eigenvalues)]) (* scale a s b)))))
  (define rotated (rotated-by spectrum))
  (check-true (bracketed? rotated (expt 10 -8) 1))
  (check-true (<= 1e4 (condition rotated) (* 1e4 (+ 1 bracket-margin))))
  (check-equal? (eigen-bottom '((1 2) (2 4))) 0)
  (match-define (cons four one) (outer-bracket '((4 0) (0 1))))
  (check-true (and (<= 4 four) (<= one 1) (= one (eigen-bottom '((4 0) (0 1))))))
  (check-equal? (norm2 '(3 4)) 5)
  (define (exactly-bracketed? G eigenvalues)
    (define-values (inner outer) (eigen-bracket G))
    (match-define (cons in-top in-bottom) inner)
    (match-define (cons out-top out-bottom) outer)
    (and (<= in-top (apply max eigenvalues) out-top)
         (<= out-bottom (apply min eigenvalues) in-bottom)))
  (define (diagonal eigenvalues)
    (for/list ([s (in-list eigenvalues)]
               [i (in-naturals)])
      (for/list ([j (in-range (length eigenvalues))])
        (if (= i j) s 0))))
  (for ([eigenvalues (in-list (list '(5) '(1 1 1) '(4 2 1/1000) (list #e1e30 1 #e1e-30) '(0 0 3)))])
    (check-true (exactly-bracketed? (diagonal eigenvalues) eigenvalues)
                (format "diagonal ~a" eigenvalues)))
  (for ([eigenvalues
         (in-list
          (list spectrum (list 1 1 (expt 10 -8)) (list 1 (expt 10 -8) (expt 10 -8)) '(0 1 2)))])
    (check-true (exactly-bracketed? (rotated-by eigenvalues) eigenvalues)
                (format "rotated ~a" eigenvalues))
    (check-true (exactly-bracketed? (rotated-by eigenvalues (expt 2 -1100))
                                    (map (lambda (s) (* s (expt 2 -1100))) eigenvalues))
                (format "rotated ~a at 2^-1100" eigenvalues)))
  (check-true (exactly-bracketed? '((1 2) (2 4)) '(0 5)))
  (define-values (singular-inner _o) (eigen-bracket '((1 2) (2 4))))
  (match-define (cons _t singular-bottom) singular-inner)
  (check-equal? singular-bottom 0)
  (check-equal? (integer-scaled '(1/2 1/3 2)) '(6 3 2 12))
  (check-equal? (integer-scaled '()) '(1))
  (check-equal? (bracket-condition '(4 . 1)) 2.0)
  (check-equal? (bracket-condition '(4 . 0)) +inf.0)
  (check-equal? (bracket-condition '(0 . 0)) +inf.0)
  (define-values (with-constant-inner _w) (eigen-bracket (diagonal '(4 0 1))))
  (check-equal? (bracket-condition with-constant-inner) +inf.0)
  (match-define (cons varying-top varying-bottom)
    (varying-inner (diagonal '(4 0 1)) with-constant-inner))
  (check-true (and (<= varying-top 4) (<= 1 varying-bottom (* 1001/1000 1))))
  (check-equal? (varying-inner (diagonal '(0 0)) '(0 . 0)) '(0 . 0))
  (check-equal? (varying-inner (diagonal '(4 1)) 'kept) 'kept))
