#lang racket/base

(require (only-in math/flonum flnext)
         (only-in racket/flonum flsingle)
         (only-in racket/list last make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-exn check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" normal-source uniform-source)
         (only-in "private/linear-data.rkt" by-dtype dtypes on-vector)
         (only-in "private/logistic-data.rkt"
                  blob-case
                  case-fit
                  coef-flat
                  default-tol-cap
                  fit-flat
                  fit-parts
                  lr-case
                  lr-case-name
                  lr-case-w
                  lr-case-X
                  lr-case-y
                  lr-fit
                  lr-fit-coef
                  lr-fit?
                  relative-gap
                  scaled-case
                  scaled-parts
                  setting
                  tight-max-iter
                  tight-tol))

(define binary (blob-case "binary" 80 3 2 11 #:weighted? #t))

(test-gpu "float64 X scaled by a power of 2 near the range's ends fits as X"
  (for* ([a (in-list '(590 -590))]
         [s (in-list (if (positive? a)
                         (list (setting 'none 1 #f) (setting 'l2 (expt 2 200) #f))
                         (list (setting 'none 1 #f))))]
         [fi (in-list '(#t #f))])
    (match-define (setting penalty C _) s)
    (test-case (format "~a ~a ~a ~a" a penalty C fi)
      (define base (scaled-case binary (expt 2 (if (positive? a) 10 -10))))
      (define (fitted c strength)
        (fit-parts
         (case-fit c 'float64 #:penalty penalty #:C (exact->inexact strength) #:fit-intercept fi)))
      (define moved
        (fitted (scaled-case base (expt 2 a))
                (if (eq? penalty 'l2)
                    (* C (expt 4 (- a)))
                    C)))
      (check-equal? moved (scaled-parts (fitted base C) (expt 2.0 (- a)))))))

(test-gpu "an l2 penalty that X times 2^-600 puts outside float64 is refused naming C"
  (check-exn #rx"C: the penalty at X's scale 2\\^-[0-9]+ is outside the float64 range"
             (lambda () (case-fit (scaled-case binary (expt 2 -600)) 'float64 #:penalty 'l2))))

(define (next-above dt v)
  ((by-dtype dt (lambda (x) (flsingle (+ x (expt 2.0 -21)))) flnext) v))

(define (with-last-column c column)
  (lr-case (lr-case-name c)
           (for/list ([row (in-list (lr-case-X c))]
                      [v (in-list column)])
             (append row (list v)))
           (lr-case-y c)
           (lr-case-w c)))

(define (last-coef parts)
  (map last (lr-fit-coef parts)))

(define last-bits-message
  #rx"^logistic-regression-fit: X: column 3 varies only in its last bits \\(spread .* of 5[.0-9]*\\); centre it$")

(test-gpu "a column that varies only in its last bits is refused, naming it"
  (for ([dt (in-list dtypes)])
    (test-case (format "~a" dt)
      (define column
        (for/list ([i (in-range (length (lr-case-y binary)))])
          (if (zero? i)
              (next-above dt 5.0)
              5.0)))
      (check-exn last-bits-message (lambda () (case-fit (with-last-column binary column) dt))))))

(define (offset-signal-case offset delta)
  (define normal (normal-source 91))
  (define uniform (uniform-source 92))
  (define rows
    (for/list ([_ (in-range 400)])
      (list (normal) (if (< (uniform) 1/2) 1 0))))
  (lr-case "offset signal"
           (for/list ([r (in-list rows)])
             (match-define (list a b) r)
             (list a (+ offset (* delta b))))
           (for/list ([r (in-list rows)])
             (match-define (list a b) r)
             (if (positive? (+ (* 3 (- b 1/2)) (* 1/2 a) (- (uniform) 1/2))) 1 0))
           #f))

(test-gpu "a signal column on a large offset is refused or fitted, never given coef 0"
  (for* ([spec (in-list '((float32 1000000 1) (float32 1000000 1/16) (float64 #e1e15 1)))]
         [penalty (in-list '(l2 none))])
    (match-define (list dt offset delta) spec)
    (test-case (format "~a ~a" spec penalty)
      (define fitted
        (with-handlers ([exn:fail? exn-message])
          (fit-parts (case-fit (offset-signal-case offset delta) dt #:penalty penalty))))
      (check-true (or (and (string? fitted) (regexp-match? #rx"varies only in its last bits" fitted))
                      (and (lr-fit? fitted) (andmap (lambda (v) (not (zero? v))) (last-coef fitted))))
                  (format "~a" fitted)))))

(test-gpu "columns whose scales put one below the dtype's normal range once staged are refused"
  (for* ([spec (in-list '((float32 -60 100) (float64 -600 500)))]
         [fi (in-list '(#t #f))])
    (match-define (list dt small large) spec)
    (test-case (format "~a ~a" spec fi)
      (define c (blob-case "mixed" 80 2 2 11 #:scales (list (expt 2 small) (expt 2 large))))
      (check-exn #rx"X: columns 0 and 1 span too wide a range; standardise them"
                 (lambda () (case-fit c dt #:penalty 'none #:fit-intercept fi))))))

(define noise
  (let ([normal (normal-source 77)])
    (for/list ([_ (in-range (length (lr-case-y binary)))])
      (/ (round (* 256 (normal))) 256))))

(define (rounded dt v)
  (by-dtype dt (flsingle (real->double-flonum v)) (real->double-flonum v)))

(define (narrow-spread dt)
  (by-dtype dt (expt 2 -10) (expt 2 -20)))

(define (centred-form-gap-bound dt)
  (by-dtype dt 1.6e-2 9.2e-6))

(test-gpu "a column of narrow relative spread is fitted as its centred form"
  (for* ([dt (in-list dtypes)]
         [penalty (in-list '(l2 none))])
    (test-case (format "~a ~a" dt penalty)
      (define spread (narrow-spread dt))
      (define (fitted column)
        (fit-parts (case-fit (with-last-column binary column)
                             dt
                             #:penalty penalty
                             #:tol (tight-tol dt)
                             #:max-iter tight-max-iter)))
      (define offset (fitted (map (lambda (z) (rounded dt (* 5 (+ 1 (* spread z))))) noise)))
      (define centred (fitted (map (lambda (z) (rounded dt (* 5 spread z))) noise)))
      (check-true (andmap (lambda (v) (not (zero? v))) (last-coef offset)))
      (define gap (relative-gap (coef-flat offset) (coef-flat centred)))
      (check-true (<= gap (centred-form-gap-bound dt)) (format "gap ~a" gap)))))

(test-gpu "an offset column is fitted within the default tol's cap of its centred form"
  (for* ([dt (in-list dtypes)]
         [penalty (in-list '(l2 none))])
    (test-case (format "~a ~a" dt penalty)
      (define (fitted column)
        (with-handlers ([exn:fail? (lambda (_) #f)])
          (fit-parts (case-fit (with-last-column binary column) dt #:penalty penalty))))
      (define offset (fitted (map (lambda (z) (+ z 16)) noise)))
      (define centred (fitted noise))
      (check-true (and offset centred #t))
      (when offset
        (define gap (relative-gap (last-coef offset) (last-coef centred)))
        (check-true (<= gap (default-tol-cap dt)) (format "gap ~a" gap))))))

(define (equal-classes n-per [order 'blocks])
  (define normal (normal-source 61))
  (define labels
    (for/list ([i (in-range (* 3 n-per))])
      (if (eq? order 'blocks)
          (quotient i n-per)
          (modulo i 3))))
  (lr-case "three equal classes"
           (for/list ([l (in-list labels)])
             (list (+ l (normal)) (- (normal) l)))
           labels
           #f))

(define balanced-gap-bound 1e-6)

(test-gpu "'balanced with uniform weights and equal classes fits as unweighted with C times w"
  (define c (equal-classes 1000))
  (for ([w (in-list '(0.1 0.3))])
    (test-case (format "~a" w)
      (define weights (on-vector (make-list (length (lr-case-y c)) w) 'float64))
      (define (balanced penalty)
        (fit-parts (case-fit c
                             'float64
                             #:sample-weight weights
                             #:class-weight 'balanced
                             #:penalty penalty
                             #:tol (tight-tol 'float64)
                             #:max-iter tight-max-iter)))
      (define (plain penalty C)
        (fit-parts (case-fit c
                             'float64
                             #:penalty penalty
                             #:C C
                             #:tol (tight-tol 'float64)
                             #:max-iter tight-max-iter)))
      (define none-gap (relative-gap (coef-flat (balanced 'none)) (coef-flat (plain 'none 1.0))))
      (check-true (<= none-gap balanced-gap-bound) (format "none gap ~a" none-gap))
      (define l2-gap (relative-gap (coef-flat (balanced 'l2)) (coef-flat (plain 'l2 w))))
      (check-true (<= l2-gap balanced-gap-bound) (format "l2 gap ~a" l2-gap)))))

(define (with-zero-rows c w)
  (lr-case (lr-case-name c)
           (append (lr-case-X c) '((0 0) (1 -1) (2 -2)))
           (append (lr-case-y c) '(0 1 2))
           (append (make-list (length (lr-case-y c)) w) '(0 0 0))))

(test-gpu "'balanced with uniform weights decides on the rows of positive weight"
  (for ([spec (in-list '((blocks 0.1) (cyclic 0.1) (blocks 0.3)))])
    (match-define (list order w) spec)
    (test-case (format "~a" spec)
      (define c (equal-classes 1000 order))
      (define (balanced weighted)
        (fit-parts (case-fit weighted
                             'float64
                             #:sample-weight (on-vector (lr-case-w weighted) 'float64)
                             #:class-weight 'balanced
                             #:tol (tight-tol 'float64)
                             #:max-iter tight-max-iter)))
      (define gap
        (relative-gap (fit-flat (balanced (with-zero-rows c w)))
                      (fit-flat (balanced (struct-copy lr-case c [w (make-list 3000 w)])))))
      (check-true (<= gap balanced-gap-bound) (format "gap ~a" gap)))))
