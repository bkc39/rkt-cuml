#lang racket/base

(require (only-in math/flonum flnext)
         (only-in racket/flonum flsingle)
         (only-in rackunit check-equal? check-exn check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" normal-source)
         (only-in "private/linear-data.rkt" dtypes on-vector)
         (only-in "private/logistic-data.rkt"
                  blob-case
                  case-fit
                  fit-parts
                  lr-case
                  lr-case-name
                  lr-case-w
                  lr-case-X
                  lr-case-y
                  relative-gap))

(define binary (blob-case "binary" 80 3 2 11 #:weighted? #t))

(define (scaled-case c factor)
  (lr-case (lr-case-name c)
           (for/list ([row (in-list (lr-case-X c))])
             (map (lambda (v) (* v factor)) row))
           (lr-case-y c)
           (lr-case-w c)))

(define (scaled-parts parts factor)
  (list (for/list ([row (in-list (car parts))])
          (map (lambda (v) (* v factor)) row))
        (cadr parts)
        (caddr parts)))

(test-gpu "float64 X times 2^600 or 2^-600 fits as X, its coef scaled by 2^-+600 (E7)"
  (for* ([a (in-list '(590 -590))]
         [setting (in-list (if (positive? a)
                               (list (list 'none 1) (list 'l2 (expt 2 200)))
                               (list (list 'none 1))))]
         [fi (in-list '(#t #f))])
    (test-case (format "~a ~a ~a" a setting fi)
      (define base (scaled-case binary (expt 2 (if (positive? a) 10 -10))))
      (define penalty (car setting))
      (define C (cadr setting))
      (define (fitted c C)
        (fit-parts (case-fit c 'float64 #:penalty penalty #:C (exact->inexact C) #:fit-intercept fi)))
      (define moved
        (fitted (scaled-case base (expt 2 a))
                (if (eq? penalty 'l2)
                    (* C (expt 4 (- a)))
                    C)))
      (check-equal? moved (scaled-parts (fitted base C) (expt 2.0 (- a)))))))

(test-gpu "an l2 penalty that X times 2^-600 puts outside float64 is refused naming C"
  (check-exn #rx"C: the penalty at X's scale 2\\^[0-9]+ is outside the float64 range"
             (lambda () (case-fit (scaled-case binary (expt 2 -600)) 'float64 #:penalty 'l2))))

(define (next-above dt v)
  (if (eq? dt 'float32)
      (flsingle (+ v (expt 2.0 -21)))
      (flnext v)))

(define (with-last-column c column)
  (lr-case (lr-case-name c)
           (for/list ([row (in-list (lr-case-X c))]
                      [v (in-list column)])
             (append row (list v)))
           (lr-case-y c)
           (lr-case-w c)))

(define (but-last row)
  (reverse (cdr (reverse row))))

(test-gpu "a column at rounding-level spread gets coef 0 and leaves the rest bitwise (E4)"
  (for ([dt (in-list dtypes)])
    (test-case (format "~a" dt)
      (define column
        (for/list ([i (in-range (length (lr-case-y binary)))])
          (if (zero? i)
              (next-above dt 5.0)
              5.0)))
      (define base (fit-parts (case-fit binary dt)))
      (define wide (fit-parts (case-fit (with-last-column binary column) dt)))
      (check-true (andmap (lambda (row) (zero? (car (reverse row)))) (car wide)))
      (check-equal? (cons (map but-last (car wide)) (cdr wide)) base))))

(define noise
  (let ([normal (normal-source 77)])
    (for/list ([_ (in-range (length (lr-case-y binary)))])
      (/ (round (* 256 (normal))) 256))))

(define (rounded dt v)
  (if (eq? dt 'float32)
      (flsingle (real->double-flonum v))
      (real->double-flonum v)))

(define (last-coef parts)
  (map (lambda (row) (car (reverse row))) (car parts)))

(test-gpu "a column of relative spread 2^-20 (float64) or 2^-10 (float32) is fitted as its centred form"
  (for* ([dt (in-list dtypes)]
         [penalty (in-list '(l2 none))])
    (test-case (format "~a ~a" dt penalty)
      (define spread
        (if (eq? dt 'float32)
            (expt 2 -10)
            (expt 2 -20)))
      (define (fitted column)
        (fit-parts (case-fit (with-last-column binary column)
                             dt
                             #:penalty penalty
                             #:tol (if (eq? dt 'float32) 1e-6 1e-10)
                             #:max-iter 100000)))
      (define offset (fitted (map (lambda (z) (rounded dt (* 5 (+ 1 (* spread z))))) noise)))
      (define centred (fitted (map (lambda (z) (rounded dt (* 5 spread z))) noise)))
      (check-true (andmap (lambda (v) (not (zero? v))) (last-coef offset)))
      (define gap (relative-gap (apply append (car offset)) (apply append (car centred))))
      (check-true (<= gap (if (eq? dt 'float32) 1.6e-2 9.2e-6)) (format "gap ~a" gap)))))

(test-gpu "an offset column is refused or fitted within S4's cap of its centred form"
  (for* ([dt (in-list dtypes)]
         [penalty (in-list '(l2 none))])
    (test-case (format "~a ~a" dt penalty)
      (define (fitted column)
        (with-handlers ([exn:fail? (lambda (_) #f)])
          (fit-parts (case-fit (with-last-column binary column) dt #:penalty penalty))))
      (define offset (fitted (map (lambda (z) (+ z 256)) noise)))
      (define centred (fitted noise))
      (when offset
        (define gap (relative-gap (last-coef offset) (last-coef centred)))
        (check-true (<= gap (if (eq? dt 'float32) 8e-2 3e-2)) (format "gap ~a" gap))))))

(define (equal-classes n-per)
  (define normal (normal-source 61))
  (define labels
    (for/list ([i (in-range (* 3 n-per))])
      (quotient i n-per)))
  (lr-case "three equal classes"
           (for/list ([l (in-list labels)])
             (list (+ l (normal)) (- (normal) l)))
           labels
           #f))

(test-gpu "'balanced with uniform weights and equal classes fits as unweighted with C times w"
  (define c (equal-classes 1000))
  (for ([w (in-list '(0.1 0.3))])
    (test-case (format "~a" w)
      (define weights (on-vector (map (lambda (_) w) (lr-case-y c)) 'float64))
      (define (flat parts)
        (apply append (car parts)))
      (define (balanced penalty)
        (fit-parts (case-fit c
                             'float64
                             #:sample-weight weights
                             #:class-weight 'balanced
                             #:penalty penalty
                             #:tol 1e-10
                             #:max-iter 100000)))
      (define (plain penalty C)
        (fit-parts (case-fit c 'float64 #:penalty penalty #:C C #:tol 1e-10 #:max-iter 100000)))
      (define none-gap (relative-gap (flat (balanced 'none)) (flat (plain 'none 1.0))))
      (check-true (<= none-gap 1e-6) (format "none gap ~a" none-gap))
      (define l2-gap (relative-gap (flat (balanced 'l2)) (flat (plain 'l2 w))))
      (check-true (<= l2-gap 1e-6) (format "l2 gap ~a" l2-gap)))))
