#lang racket/base

(require (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-result-coef
                  linear-regression-result-intercept)
         (only-in racket/list append-map make-list)
         (only-in racket/match match-define)
         (only-in rackunit test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt"
                  algorithms
                  check-law
                  dtypes
                  lcase
                  lcase-X
                  lcase-name
                  lcase-w
                  lcase-y
                  max-abs
                  normal-case
                  on-matrix
                  on-vector
                  oracle-lstsq
                  rounded-case
                  solution
                  solution-kappa
                  unit-roundoff))

(define cases
  (list (normal-case 200 7 0)
        (normal-case 200 7 1 #:weighted? #t)
        (normal-case 200 2 2 #:weighted? #t)
        (normal-case 20 7 3)))

(define (fit-of X y w dt a fi)
  (define r
    (linear-regression-fit (on-matrix X dt)
                           (on-vector y dt)
                           #:sample-weight (on-vector w dt)
                           #:algorithm a
                           #:fit-intercept fi))
  (list (map inexact->exact (device-vector->list (linear-regression-result-coef r)))
        (inexact->exact (linear-regression-result-intercept r))))

(define (gap a b)
  (max-abs (map - (cons (cadr a) (car a)) (cons (cadr b) (car b)))))

(define (size-of f)
  (max 1 (abs (cadr f)) (max-abs (car f))))

(define (kappa-of c fi)
  (solution-kappa (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) fi)))

(define (weights-of c)
  (or (lcase-w c) (make-list (length (lcase-y c)) 1)))

(define (permuted xs)
  (define v (list->vector xs))
  (define n (vector-length v))
  (for/list ([i (in-range n)])
    (vector-ref v (modulo (* i 7) n))))

(define (each-case proc)
  (for* ([c (in-list cases)]
         [fi (in-list '(#t #f))])
    (define kappa (kappa-of c fi))
    (for* ([dt (in-list dtypes)]
           [a (in-list algorithms)])
      (test-case (format "~a ~a ~a ~a" (lcase-name c) fi dt a)
        (define u (unit-roundoff dt))
        (proc c fi dt a u kappa (fit-of (lcase-X c) (lcase-y c) (lcase-w c) dt a fi))))))

(test-gpu "T5-T8: equivalent weightings and orderings agree within the bound"
  (each-case
   (lambda (c fi dt a u kappa fitted)
     (define X (lcase-X c))
     (define y (lcase-y c))
     (define w (weights-of c))
     (define counts
       (for/list ([i (in-range (length y))])
         (add1 (modulo (* i 5) 4))))
     (define repeated
       (fit-of (append-map (lambda (row k) (make-list k row)) X counts)
               (append-map (lambda (v k) (make-list k v)) y counts)
               #f
               dt
               a
               fi))
     (check-law "T5" (gap (fit-of X y counts dt a fi) repeated) (* 32 u kappa (size-of repeated)))
     (define gone '(3 50 101))
     (define (kept xs)
       (for/list ([x (in-list xs)]
                  [i (in-naturals)]
                  #:unless (memv i gone))
         x))
     (define zeroed
       (for/list ([v (in-list w)]
                  [i (in-naturals)])
         (if (memv i gone) 0 v)))
     (when (> (length y) 101)
       (check-law "T6"
                  (gap (fit-of X y zeroed dt a fi) (fit-of (kept X) (kept y) (kept w) dt a fi))
                  (* 32 u kappa (size-of fitted))))
     (check-law "T7"
                (gap (fit-of X y (map (lambda (v) (* 3 v)) w) dt a fi) (fit-of X y w dt a fi))
                (* 32 u kappa (size-of fitted)))
     (check-law "T8"
                (gap (fit-of (permuted X) (permuted y) (and (lcase-w c) (permuted w)) dt a fi) fitted)
                (* 32 u kappa (size-of fitted))))))

(test-gpu "T9-T11: column order, a shifted column and a shifted y move the fit as they should"
  (each-case
   (lambda (c fi dt a u kappa fitted)
     (define X (lcase-X c))
     (define y (lcase-y c))
     (define w (lcase-w c))
     (match-define (list coef b) fitted)
     (define reversed (fit-of (map reverse X) y w dt a fi))
     (check-law "T9"
                (gap (list (reverse (car reversed)) (cadr reversed)) fitted)
                (* 128 u kappa (size-of fitted)))
     (when fi
       (define shifted (fit-of (map (lambda (row) (cons (+ 2 (car row)) (cdr row))) X) y w dt a fi))
       (check-law "T10"
                  (gap shifted (list coef (- b (* 2 (car coef)))))
                  (* 128 u kappa (size-of fitted)))
       (define lifted (fit-of X (map (lambda (v) (+ v 5)) y) w dt a fi))
       (check-law "T11" (gap lifted (list coef (+ b 5))) (* 8 u kappa (max (max-abs y) 5)))))))

(define (wide-case)
  (normal-case 12 30 6))

(define (row-kappa rows)
  (define G
    (for/list ([r (in-list rows)])
      (for/list ([s (in-list rows)])
        (for/sum ([a (in-list r)] [b (in-list s)]) (* a b)))))
  (sqrt (solution-kappa (oracle-lstsq G (make-list (length rows) 0) #f #f))))

(test-gpu "T13: wide X is min-norm in float64 and interpolates in float32"
  (define c (wide-case))
  (define X (lcase-X c))
  (define y (lcase-y c))
  (match-define (solution c* _ _) (oracle-lstsq X y #f #f))
  (define kappa (row-kappa X))
  (for ([a (in-list algorithms)])
    (test-case (format "~a" a)
      (define f64 (fit-of X y #f 'float64 a #f))
      (check-law "T13 float64"
                 (max-abs (map - (car f64) c*))
                 (* 64 (unit-roundoff 'float64) kappa (max 1 (max-abs c*))))
      (match-define (list coef b) (fit-of X y #f 'float32 a #t))
      (define residual
        (for/list ([row (in-list X)]
                   [yi (in-list y)])
          (- (+ b (for/sum ([x (in-list row)] [cj (in-list coef)]) (* x cj))) yi)))
      (check-law "T13 float32" (max-abs residual) (* (expt 2 -12) (max-abs y))))))

(define (scaled-columns c factors y-factor)
  (lcase (lcase-name c)
         (for/list ([row (in-list (lcase-X c))])
           (for/list ([x (in-list row)]
                      [s (in-list factors)])
             (* x s)))
         (map (lambda (v) (* v y-factor)) (lcase-y c))
         (lcase-w c)))

(define (reach-fixtures dt)
  (define column-scales
    (if (eq? dt 'float32)
        '(#e1e-5 #e1e-7 #e1e30)
        '(#e1e150 #e1e-150)))
  (append (for/list ([s (in-list column-scales)])
            (list (format "column 1 at ~a" s) (list 1 s 1 1 1 1 1) 1))
          (list (list "X at 1e-12" (make-list 7 #e1e-12) 1)
                (list "y at 1e37" (make-list 7 1) #e1e37)
                (list "subnormal y" (make-list 7 1) #e1e-40))))

(test-gpu "T14: the prescale reaches columns, X and y far from 1"
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
        (define coef (car (fit-of (lcase-X c) (lcase-y c) (lcase-w c) dt a fi)))
        (check-law "T14"
                   (max-abs (map - (staged coef) (staged c*)))
                   (* 128 (unit-roundoff dt) kappa kappa (max 1 (max-abs (staged c*)))))))))
