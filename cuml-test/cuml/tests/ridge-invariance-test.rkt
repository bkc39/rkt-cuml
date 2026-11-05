#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt"
                  dtypes
                  insert-column
                  kept-rows
                  lcase
                  lcase-name
                  lcase-w
                  lcase-X
                  lcase-y
                  normal-case
                  remove-at
                  scaled-case
                  zeroed)
         (only-in "private/ridge-data.rkt" ridge-case-fit solvers)
         (only-in "private/scaler-data.rkt" bits-equal?))

(define base (normal-case 200 7 3 #:weighted? #t))

(define (shifts dt)
  (if (eq? dt 'float32)
      '(-12 -3 5 12)
      '(-30 -3 5 30)))

(test-gpu "X times 2^a with alpha times 4^a gives 2^-a times the coefficients, bitwise (B2)"
  (for* ([dt (in-list dtypes)]
         [s (in-list solvers)]
         [fi (in-list '(#t #f))]
         [a (in-list (shifts dt))])
    (test-case (format "~a ~a ~a 2^~a" dt s fi a)
      (match-define (list coef b) (ridge-case-fit base dt #:alpha 3/2 #:solver s #:fit-intercept fi))
      (check-true (bits-equal? (ridge-case-fit (scaled-case base #:X (expt 2 a))
                                               dt
                                               #:alpha (* 3/2 (expt 4 a))
                                               #:solver s
                                               #:fit-intercept fi)
                               (list (map (lambda (x) (* x (expt 2 (- a)))) coef) b))))))

(test-gpu "y times 2^c gives 2^c times the fit, bitwise (B3)"
  (for* ([dt (in-list dtypes)]
         [s (in-list solvers)]
         [fi (in-list '(#t #f))]
         [c (in-list '(-40 7 40))])
    (test-case (format "~a ~a ~a 2^~a" dt s fi c)
      (match-define (list coef b) (ridge-case-fit base dt #:alpha 3/2 #:solver s #:fit-intercept fi))
      (check-true (bits-equal? (ridge-case-fit (scaled-case base #:y (expt 2 c))
                                               dt
                                               #:alpha 3/2
                                               #:solver s
                                               #:fit-intercept fi)
                               (list (map (lambda (x) (* x (expt 2 c))) coef) (* (expt 2 c) b)))))))

(test-gpu "weights times 4^m with alpha times 4^m give the same fit, bitwise (B4)"
  (for* ([dt (in-list dtypes)]
         [s (in-list solvers)]
         [fi (in-list '(#t #f))]
         [m (in-list '(-3 2))])
    (test-case (format "~a ~a ~a 4^~a" dt s fi m)
      (check-true (bits-equal?
                   (ridge-case-fit (scaled-case base #:w (expt 4 m))
                                   dt
                                   #:alpha (* 3/2 (expt 4 m))
                                   #:solver s
                                   #:fit-intercept fi)
                   (ridge-case-fit base dt #:alpha 3/2 #:solver s #:fit-intercept fi))))))

(define (deleted-cases fi)
  (append (list (normal-case 200 7 6 #:weighted? #t) (normal-case 1000 7 7 #:weighted? #t))
          (if fi
              '()
              (list (normal-case 5000 7 8 #:weighted? #t)))))

(test-gpu "zero weight on three rows is those rows deleted, bitwise (B5)"
  (for* ([fi (in-list '(#t #f))]
         [c (in-list (deleted-cases fi))]
         [dt (in-list dtypes)]
         [s (in-list solvers)])
    (test-case (format "~a ~a ~a ~a" (lcase-name c) fi dt s)
      (define n (length (lcase-y c)))
      (define gone (list 0 (quotient n 2) (sub1 n)))
      (define (kept xs)
        (kept-rows xs gone))
      (check-true (bits-equal?
                   (ridge-case-fit (lcase "" (lcase-X c) (lcase-y c) (zeroed (lcase-w c) gone))
                                   dt
                                   #:alpha 5
                                   #:solver s
                                   #:fit-intercept fi)
                   (ridge-case-fit (lcase "" (kept (lcase-X c)) (kept (lcase-y c)) (kept (lcase-w c)))
                                   dt
                                   #:alpha 5
                                   #:solver s
                                   #:fit-intercept fi))))))

(test-gpu "a constant (or, without an intercept, zero) column gets 0.0 and moves nothing (B6)"
  (for* ([fi (in-list '(#t #f))]
         [c (in-list (list base (normal-case 2000 7 9)))]
         [dt (in-list dtypes)]
         [s (in-list solvers)]
         [at (in-list '(0 3 7))])
    (test-case (format "~a ~a ~a ~a at ~a" (lcase-name c) fi dt s at)
      (match-define (list coef b) (ridge-case-fit c dt #:alpha 1/2 #:solver s #:fit-intercept fi))
      (define value (if fi 5/4 0))
      (match-define (list coef+ b+)
        (ridge-case-fit
         (lcase "" (insert-column (lcase-X c) at (lambda (_) value)) (lcase-y c) (lcase-w c))
         dt
         #:alpha 1/2
         #:solver s
         #:fit-intercept fi))
      (check-equal? (list-ref coef+ at) 0.0)
      (check-true (bits-equal? (list (remove-at coef+ at) b+) (list coef b))))))
