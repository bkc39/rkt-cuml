#lang racket/base

(require (only-in racket/list append-map make-list take)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt"
                  algorithms
                  check-law
                  dtypes
                  exact-fit
                  fit-gap
                  fit-size
                  gap
                  kappa-of
                  kept-rows
                  lcase
                  lcase-X
                  lcase-name
                  lcase-w
                  lcase-y
                  fit-of
                  max-abs
                  normal-case
                  oracle-lstsq
                  power-spread-case
                  power-spread-gone
                  reach-fixtures
                  rounded-case
                  scaled-columns
                  solution
                  solution-kappa
                  unit-roundoff
                  weights-or-ones
                  zeroed)
         (only-in "private/scaler-data.rkt" bits-equal?))

(define cases
  (list (normal-case 200 7 0)
        (normal-case 200 7 1 #:weighted? #t)
        (normal-case 200 2 2 #:weighted? #t)
        (normal-case 20 7 3)))

(define (fit X y w dt a fi)
  (exact-fit (fit-of X y w dt #:algorithm a #:fit-intercept fi)))

(define (permuted xs)
  (define v (list->vector xs))
  (define n (vector-length v))
  (for/list ([i (in-range n)])
    (vector-ref v (modulo (* i 7) n))))

(struct trial (c fi dt a kappa fitted))

(define (each-case proc)
  (for* ([c (in-list cases)]
         [fi (in-list '(#t #f))])
    (define kappa (kappa-of c fi))
    (for* ([dt (in-list dtypes)]
           [a (in-list algorithms)])
      (test-case (format "~a ~a ~a ~a" (lcase-name c) fi dt a)
        (proc (trial c fi dt a kappa (fit (lcase-X c) (lcase-y c) (lcase-w c) dt a fi)))))))

(test-gpu "equivalent weightings and orderings agree within the bound"
  (each-case
   (lambda (t)
     (match-define (trial c fi dt a kappa fitted) t)
     (define u (unit-roundoff dt))
     (define X (lcase-X c))
     (define y (lcase-y c))
     (define w (weights-or-ones (lcase-w c) y))
     (define counts
       (for/list ([i (in-range (length y))])
         (add1 (modulo (* i 5) 4))))
     (define repeated
       (fit (append-map (lambda (row k) (make-list k row)) X counts)
            (append-map (lambda (v k) (make-list k v)) y counts)
            #f
            dt
            a
            fi))
     (check-law "T5" (fit-gap (fit X y counts dt a fi) repeated) (* 32 u kappa (fit-size repeated)))
     (define gone '(3 50 101))
     (define (kept xs)
       (kept-rows xs gone))
     (when (> (length y) 101)
       (check-true
        (bits-equal? (fit-of X y (zeroed w gone) dt #:algorithm a #:fit-intercept fi)
                     (fit-of (kept X) (kept y) (kept w) dt #:algorithm a #:fit-intercept fi))
        "a zero weight is a deleted row"))
     (check-law "T7"
                (fit-gap (fit X y (map (lambda (v) (* 3 v)) w) dt a fi) (fit X y w dt a fi))
                (* 32 u kappa (fit-size fitted)))
     (check-law "T8"
                (fit-gap (fit (permuted X) (permuted y) (and (lcase-w c) (permuted w)) dt a fi)
                         fitted)
                (* 32 u kappa (fit-size fitted))))))

(test-gpu "zero weights delete rows bitwise when a column's weighted spread is within ulps of a power of two"
  (for* ([step (in-range -10 26 2)]
         #:do [(define c (power-spread-case 1 step)) (define w (lcase-w c))]
         [a (in-list algorithms)])
    (test-case (format "~a ~a" (lcase-name c) a)
      (define (kept xs)
        (kept-rows xs power-spread-gone))
      (check-true
       (bits-equal? (fit-of (lcase-X c) (lcase-y c) (zeroed w power-spread-gone) 'float64 #:algorithm a)
                    (fit-of (kept (lcase-X c)) (kept (lcase-y c)) (kept w) 'float64 #:algorithm a))
       "a zero weight is a deleted row"))))

(test-gpu "column order, a shifted column and a shifted y move the fit as they should"
  (each-case
   (lambda (t)
     (match-define (trial c fi dt a kappa fitted) t)
     (define u (unit-roundoff dt))
     (define X (lcase-X c))
     (define y (lcase-y c))
     (define w (lcase-w c))
     (match-define (list coef b) fitted)
     (match-define (list reversed-coef reversed-b) (fit (map reverse X) y w dt a fi))
     (check-law "T9"
                (fit-gap (list (reverse reversed-coef) reversed-b) fitted)
                (* 128 u kappa (fit-size fitted)))
     (when fi
       (define shifted
         (fit (for/list ([row (in-list X)])
                (match-define (cons x0 rest) row)
                (cons (+ 2 x0) rest))
              y
              w
              dt
              a
              fi))
       (match-define (cons c0 _) coef)
       (check-law "T10"
                  (fit-gap shifted (list coef (- b (* 2 c0))))
                  (* 128 u kappa (fit-size fitted)))
       (define lifted (fit X (map (lambda (v) (+ v 5)) y) w dt a fi))
       (check-law "T11" (fit-gap lifted (list coef (+ b 5))) (* 8 u kappa (max (max-abs y) 5)))))))

(define (predicted-by f X)
  (match-define (list coef b) (exact-fit f))
  (for/list ([row (in-list X)])
    (+ b (for/sum ([x (in-list row)] [cj (in-list coef)]) (* x cj)))))

(define (wide-case)
  (normal-case 12 30 6))

(define (row-kappa rows)
  (define G
    (for/list ([r (in-list rows)])
      (for/list ([s (in-list rows)])
        (for/sum ([a (in-list r)] [b (in-list s)]) (* a b)))))
  (sqrt (solution-kappa (oracle-lstsq G (make-list (length rows) 0) #f #f))))

(test-gpu "wide X is min-norm in float64 and interpolates in float32"
  (define c (wide-case))
  (define X (lcase-X c))
  (define y (lcase-y c))
  (match-define (solution c* _ _) (oracle-lstsq X y #f #f))
  (define kappa (row-kappa X))
  (for ([a (in-list algorithms)])
    (test-case (format "~a" a)
      (match-define (list f64 _) (fit X y #f 'float64 a #f))
      (check-law "T13 float64"
                 (gap f64 c*)
                 (* 64 (unit-roundoff 'float64) kappa (max 1 (max-abs c*))))
      (check-law "T13 float32"
                 (max-abs (map - (predicted-by (fit X y #f 'float32 a #t) X) y))
                 (* (expt 2 -12) (max-abs y))))))

(test-gpu "the prescale reaches columns, X and y far from 1"
  (define base (normal-case 200 7 0 #:weighted? #t))
  (for* ([dt (in-list dtypes)]
         [fi (in-list '(#t #f))]
         [fixture (in-list (reach-fixtures dt))])
    (match-define (list name factors y-factor) fixture)
    (define c (rounded-case (scaled-columns base factors y-factor) dt))
    (match-define (solution c* _ _) (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) fi))
    (define kappa (kappa-of base fi))
    (define (staged coef)
      (for/list ([x (in-list coef)]
                 [s (in-list factors)])
        (/ (* x s) y-factor)))
    (for ([a (in-list algorithms)])
      (test-case (format "~a ~a ~a ~a" name dt fi a)
        (match-define (list coef _) (fit (lcase-X c) (lcase-y c) (lcase-w c) dt a fi))
        (check-law "T14"
                   (max-abs (map - (staged coef) (staged c*)))
                   (* 128 (unit-roundoff dt) kappa kappa (max 1 (max-abs (staged c*)))))))))

(define short-fixtures
  (let ([square (normal-case 8 4 1)])
    (list
     (list "a square X with an intercept" (take (lcase-X square) 4) (take (lcase-y square) 4) #t)
     (list "mixed column scales with an intercept" '((1 0 0) (0 1024 0) (0 0 1/1024)) '(1 2 3) #t)
     (list "four columns on three rows" '((1 0 0 1) (0 1024 0 1) (0 0 1/1024 1)) '(1 2 3) #f))))

(define junk-rows '((3 5 7 9) (2 4 6 8) (1 1 2 2)))

(test-gpu "with too few positive-weight rows the fit is the minimum-norm one, and zero weights delete rows"
  (for* ([fixture (in-list short-fixtures)]
         #:do [(match-define (list name X y fi) fixture)
               (match-define (solution c* b* _) (oracle-lstsq X y #f fi))
               (define d (length c*))
               (define ones (make-list (length y) 1))
               (define padded-X (append X (map (lambda (row) (take row d)) junk-rows)))
               (define padded-y (append y '(4 5 6)))
               (define padded-w (append ones '(0 0 0)))]
         [dt (in-list dtypes)]
         [a (in-list algorithms)])
    (test-case (format "~a ~a ~a" name dt a)
      (define fitted (fit-of X y ones dt #:algorithm a #:fit-intercept fi))
      (if (and fi (eq? dt 'float32))
          (check-law "interpolation"
                     (max-abs (map - (predicted-by fitted X) y))
                     (* (expt 2 -12) (max-abs y)))
          (check-law "minimum norm"
                     (fit-gap fitted (list c* b*))
                     (* 64 (unit-roundoff dt) (fit-size (list c* b*)))))
      (check-true (bits-equal? (fit-of padded-X padded-y padded-w dt #:algorithm a #:fit-intercept fi)
                               fitted)
                  "zero-weight rows are deleted rows"))))

(test-gpu "a column the shared power underflows gets 0.0, not a refusal"
  (for* ([dt (in-list dtypes)]
         #:do
         [(define big (if (eq? dt 'float32) #e1e30 #e1e200))
          (define tiny (if (eq? dt 'float32) #e1e-20 #e1e-200))
          (define c
            (rounded-case (lcase "underflow" `((,big ,tiny 1) (,(* 2 big) ,(* 3 tiny) 5)) '(1 2) #f)
                          dt))
          (match-define (solution c* b* _) (oracle-lstsq (lcase-X c) (lcase-y c) #f #t))]
         [a (in-list algorithms)])
    (test-case (format "~a ~a" dt a)
      (match-define (list coef b) (fit-of (lcase-X c) (lcase-y c) #f dt #:algorithm a))
      (match-define (list _ underflowed _) coef)
      (check-equal? underflowed 0.0)
      (check-law "coef" (gap (map inexact->exact coef) c*) (* 64 (unit-roundoff dt) (max-abs c*)))
      (check-law "intercept" (abs (- (inexact->exact b) b*)) (* 8 (unit-roundoff dt) 2)))))

(define (vanishing-fixtures big tiny)
  (list (list "square" '((0 0) (1 0) (0 1)) '(1 2 4) (list big tiny tiny) #t)
        (list "tall, nothing left to solve"
              '((0 0) (0 0) (1 0) (0 1) (2 0) (0 2))
              '(1 1 2 4 3 7)
              (list big big tiny tiny tiny tiny)
              #t)
        (list "tall, solved"
              '((0 0) (1 0) (0 1) (1 1) (2 0) (0 2))
              '(1 2 4 6 3 7)
              (list big big big big tiny tiny)
              #t)
        (list "no intercept" '((1 0) (1 1) (2 1)) '(1 3 4) (list big tiny tiny) #f)))

(test-gpu "a positive weight that stages to 0 deletes its row"
  (for* ([dt (in-list dtypes)]
         #:do [(define-values (big tiny)
                 (if (eq? dt 'float32)
                     (values 1e30 1e-20)
                     (values 1e300 1e-30)))]
         [fixture (in-list (vanishing-fixtures big tiny))]
         #:do [(match-define (list name X y w fi) fixture)
               (define dropped (map (lambda (v) (if (= v tiny) 0 v)) w))]
         [a (in-list algorithms)])
    (test-case (format "~a ~a ~a" name dt a)
      (check-true (bits-equal? (fit-of X y w dt #:algorithm a #:fit-intercept fi)
                               (fit-of X y dropped dt #:algorithm a #:fit-intercept fi))))))
