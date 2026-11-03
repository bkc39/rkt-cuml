#lang racket/base

(require (only-in cuml/metrics accuracy-score mean-squared-error r2-score)
         (only-in racket/flonum flsingle)
         (only-in racket/list append* drop-right last make-list)
         (only-in racket/match match-define)
         (only-in racket/math exact-floor sqr)
         (only-in rackunit check check-equal? check-true test-case)
         (only-in raft list*->device-matrix list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" normal-source uniform-source)
         (only-in "private/metrics-data.rkt"
                  exact-accuracy
                  exact-mse
                  exact-r2
                  mean
                  pow2-scaled
                  regression-cases)
         (only-in "private/scaler-data.rkt" rows-of))

(define (vec xs dt)
  (list->device-vector xs #:dtype dt))

(define (rows xs dt)
  (list*->device-matrix xs #:dtype dt))

(define (weights-of w [dt 'float64])
  (and w (vec w dt)))

(define (r2 y p [w #f] #:dtype [dt 'float64] #:pred [pt dt] #:weight [wt 'float64])
  (r2-score (vec y dt) (vec p pt) #:sample-weight (weights-of w wt)))

(define (mse y p [w #f] #:dtype [dt 'float64] #:pred [pt dt] #:weight [wt 'float64])
  (define (on xs t)
    (if (list? (car xs))
        (rows xs t)
        (vec xs t)))
  (mean-squared-error (on y dt) (on p pt) #:sample-weight (weights-of w wt)))

(define (accuracy a b [w #f] #:true [at 'int32] #:pred [bt at] #:weight [wt 'float64])
  (accuracy-score (vec a at) (vec b bt) #:sample-weight (weights-of w wt)))

(define (r2-gap ours exact)
  (/ (abs (- (inexact->exact ours) exact)) (max 1 (abs (- 1 exact)))))

(define (relative-gap ours exact)
  (/ (abs (- (inexact->exact ours) exact)) exact))

(define int32-palette '(-1 0 1 2 7 2147483647 -2147483647 -2147483648))
(define int64-palette (append int32-palette (list (expt 2 40) (- (expt 2 40)) (expt 2 32))))

(define (palette dt)
  (if (eq? dt 'int32) int32-palette int64-palette))

(define (label-cases)
  (define next (uniform-source 860))
  (define (pick xs)
    (list-ref xs (exact-floor (* (next) (length xs)))))
  (for/list ([i (in-range 200)])
    (match-define (list at bt) (list-ref '((int32 int32) (int32 int64) (int64 int64)) (modulo i 3)))
    (define n (add1 (exact-floor (* (next) 5000))))
    (define truth
      (for/list ([_ (in-range n)])
        (pick (palette at))))
    (define found
      (for/list ([t (in-list truth)])
        (define u (next))
        (cond
          [(and (eq? bt 'int64) (eqv? t 0) (< u 0.2)) (expt 2 32)]
          [(< u 0.7) t]
          [else (pick (palette bt))])))
    (list at bt truth found)))

(test-gpu "A1: unweighted accuracy is the exact count over n, correctly rounded"
  (for ([c (in-list (label-cases))]
        [i (in-naturals)])
    (match-define (list at bt truth found) c)
    (test-case (format "case ~a ~a x ~a" i at bt)
      (check-equal? (accuracy truth found #:true at #:pred bt)
                    (exact->inexact (exact-accuracy truth found))))))

(define (integer-weights n seed)
  (define next (uniform-source seed))
  (define w
    (for/list ([_ (in-range n)])
      (exact->inexact (exact-floor (* 9 (next))))))
  (if (andmap zero? w)
      (cons 1.0 (cdr w))
      w))

(test-gpu "A2: accuracy is symmetric bit for bit, with and without weights"
  (for ([c (in-list (label-cases))]
        [i (in-range 40)])
    (match-define (list at bt truth found) c)
    (define w (integer-weights (length truth) i))
    (define u (map (lambda (x) (+ 0.25 x)) w))
    (check-equal? (accuracy truth found #:true at #:pred bt)
                  (accuracy found truth #:true bt #:pred at))
    (check-equal? (accuracy truth found u #:true at #:pred bt)
                  (accuracy found truth u #:true bt #:pred at))))

(test-gpu "A3: identical labels score 1.0; labels that differ everywhere score 0.0, weighted too"
  (define ys '(3 -1 7 7 2147483647))
  (check-equal? (accuracy ys ys) 1.0)
  (check-equal? (accuracy ys ys '(0.0 1.0 2.0 0.5 3.0)) 1.0)
  (check-equal? (accuracy ys (map add1 ys) #:true 'int64) 0.0)
  (check-equal? (accuracy ys (map add1 ys) '(0.0 1.0 2.0 0.5 3.0) #:true 'int64) 0.0))

(define (repeated xs counts)
  (append* (for/list ([x (in-list xs)]
                      [c (in-list counts)])
             (make-list (inexact->exact c) x))))

(test-gpu "A4: integer weights equal the unweighted score on repeated samples, bit for bit"
  (for ([c (in-list (label-cases))]
        [i (in-range 60)])
    (match-define (list at bt truth found) c)
    (define w (integer-weights (length truth) (+ 100 i)))
    (test-case (format "case ~a" i)
      (check-equal? (accuracy truth found w #:true at #:pred bt)
                    (accuracy (repeated truth w) (repeated found w) #:true at #:pred bt)))))

(test-gpu "R1: r2 of a non-constant y against itself is 1.0"
  (for* ([dt (in-list '(float32 float64))]
         [w (in-list (list #f '(1.0 0.0 2.5 0.125)))])
    (define y '(3.0 -0.5 2.0 7.0))
    (check-equal? (r2 y y w #:dtype dt) 1.0)))

(test-gpu "R2, R5: r2 is within 1e-14 of the exact rational value and at most 1.0"
  (define worst
    (for/fold ([worst 0]) ([c (in-list (regression-cases))])
      (match-define (list i dt y p w) c)
      (define ours (r2 y p w #:dtype dt))
      (define gap (r2-gap ours (exact-r2 y p w)))
      (test-case (format "case ~a ~a" i dt)
        (check-true (<= ours 1.0) "R2")
        (check-true (<= gap 1e-14) (format "R5: ~a off the exact value" (exact->inexact gap))))
      (max worst gap)))
  (printf "R5: worst r2 gap ~a\n" (exact->inexact worst)))

(define (constant-cases)
  (for*/list ([dt (in-list '(float32 float64))]
              [y (in-list (list '(2.0 2.0 2.0)
                                (make-list 3 0.1)
                                (make-list 1000 0.1)
                                (make-list 7 (/ 1.0 3.0))))])
    (list dt y (append (drop-right y 1) (list (+ (last y) 1e-3))))))

(test-gpu "R3: on a constant y-true r2 is 1.0 for a perfect prediction and 0.0 otherwise"
  (for ([c (in-list (constant-cases))])
    (match-define (list dt y off) c)
    (test-case (format "~a ~a x ~a" dt (car y) (length y))
      (check-equal? (r2 y y #:dtype dt) 1.0)
      (check-equal? (r2 y off #:dtype dt) 0.0)))
  (check-equal? (r2 '(1.0) '(1.0)) 1.0)
  (check-equal? (r2 '(1.0) '(2.0)) 0.0)
  (check-equal? (r2 '(5.0 1.0 5.0) '(5.0 9.0 5.0) '(1.0 0.0 2.0)) 1.0)
  (check-equal? (r2 '(5.0 1.0 5.0) '(5.0 1.0 6.0) '(1.0 0.0 2.0)) 0.0))

(test-gpu "M1: the error of y against itself is 0.0, and the error is symmetric bit for bit"
  (for ([c (in-list (regression-cases))]
        [_ (in-range 30)])
    (match-define (list _ dt y p w) c)
    (check-equal? (mse y y w #:dtype dt) 0.0)
    (check-equal? (mse y p w #:dtype dt) (mse p y w #:dtype dt))))

(define (integer-values count seed)
  (define next (uniform-source seed))
  (for/list ([_ (in-range count)])
    (- (exact-floor (* (next) 2049)) 1024)))

(test-gpu "M2: on integer values the error is the exact sum over n·d, correctly rounded"
  (for* ([dt (in-list '(float32 float64))]
         [shape (in-list '((1 1) (7 1) (4096 1) (5 3) (1024 4) (64 64)))]
         [weighted? (in-list '(#f #t))])
    (match-define (list n d) shape)
    (define seed (+ n d (if weighted? 1 0)))
    (define (data k)
      (define xs (map exact->inexact (integer-values (* n d) (+ seed k))))
      (if (= d 1)
          xs
          (rows-of xs d)))
    (define y (data 0))
    (define p (data 1))
    (define w (and weighted? (integer-weights n seed)))
    (test-case (format "~a ~ax~a ~a" dt n d (if weighted? "weighted" ""))
      (check-equal? (mse y p w #:dtype dt) (exact->inexact (exact-mse y p w))))))

(define (normal-rows n d seed)
  (define next (normal-source seed))
  (for/list ([_ (in-range n)])
    (for/list ([j (in-range d)])
      (+ (* 10.0 j) (next)))))

(define (column rows j)
  (map (lambda (r) (list-ref r j)) rows))

(test-gpu "M3: a matrix's error is within 1e-14 of the mean of its columns' errors"
  (for* ([dt (in-list '(float32 float64))]
         [weighted? (in-list '(#f #t))]
         [d (in-list '(2 3 5))])
    (define y (normal-rows 300 d (+ d 10)))
    (define p (normal-rows 300 d (+ d 20)))
    (define w (and weighted? (integer-weights 300 d)))
    (define whole (mse y p w #:dtype dt))
    (define columns
      (mean (for/list ([j (in-range d)])
              (mse (column y j) (column p j) w #:dtype dt))))
    (check-true (<= (abs (- whole columns)) (* 1e-14 columns))
                (format "~a against ~a" whole columns))))

(test-gpu "M4: the error is finite when only the float32 square overflows, inf past the flonum range"
  (define big (flsingle 1e19))
  (check-equal? (mse (list big 0.0) (list (- big) 0.0) #:dtype 'float32)
                (exact->inexact (/ (sqr (* 2 (inexact->exact big))) 2)))
  (check-equal? (mse (list (expt 2.0 600) 0.0) '(0.0 0.0)) +inf.0))

(test-gpu "M5: the error is within 1e-14 of the exact rational value"
  (define worst
    (for/fold ([worst 0]) ([c (in-list (regression-cases))])
      (match-define (list i dt y p w) c)
      (define gap (relative-gap (mse y p w #:dtype dt) (exact-mse y p w)))
      (test-case (format "case ~a ~a" i dt)
        (check-true (<= gap 1e-14) (format "~a off the exact value" (exact->inexact gap))))
      (max worst gap)))
  (printf "M5: worst error gap ~a\n" (exact->inexact worst)))

(define (sample-cases)
  (for/list ([c (in-list (regression-cases))]
             #:when (zero? (modulo (car c) 10)))
    c))

(test-gpu "X3: scaling both arguments by 2^k keeps r2 and scales the error by 2^2k, exactly"
  (for ([c (in-list (sample-cases))])
    (match-define (list i dt y p w) c)
    (define base-r2 (r2 y p w #:dtype dt))
    (define base-mse (mse y p w #:dtype dt))
    (test-case (format "case ~a ~a" i dt)
      (for ([k (in-list (if (eq? dt 'float32)
                            '(-60 60)
                            '(-600 600)))])
        (check-equal? (r2 (pow2-scaled y k) (pow2-scaled p k) w #:dtype dt) base-r2))
      (for ([k (in-list (if (eq? dt 'float32)
                            '(-60 60)
                            '(-250 250)))])
        (check-equal? (mse (pow2-scaled y k) (pow2-scaled p k) w #:dtype dt)
                      (* base-mse (expt 2 (* 2 k))))))))

(test-gpu "X4: scaling the weights by 2^k leaves all three scores unchanged, bit for bit"
  (define-values (truth found) (values '(0 1 2 2 3 1 0 4) '(0 1 1 2 3 0 0 4)))
  (for ([c (in-list (sample-cases))])
    (match-define (list i dt y p _) c)
    (define w (integer-weights (length y) i))
    (define v (integer-weights (length truth) (+ i 1)))
    (for* ([wt (in-list '(float32 float64))]
           [k (in-list (if (eq? wt 'float32)
                           '(-100 100)
                           '(-1003 1020)))])
      (test-case (format "case ~a ~a weights ~a scaled 2^~a" i dt wt k)
        (check-equal? (r2 y p (pow2-scaled w k) #:dtype dt #:weight wt)
                      (r2 y p w #:dtype dt #:weight wt))
        (check-equal? (mse y p (pow2-scaled w k) #:dtype dt #:weight wt)
                      (mse y p w #:dtype dt #:weight wt))
        (check-equal? (accuracy truth found (pow2-scaled v k) #:weight wt)
                      (accuracy truth found v #:weight wt))))))

(test-gpu "X5: all-ones weights give the same bits as no weights"
  (define-values (truth found) (values '(0 1 2 2 3 1 0 4) '(0 1 1 2 3 0 0 4)))
  (for ([c (in-list (sample-cases))])
    (match-define (list _ dt y p _) c)
    (define ones (make-list (length y) 1.0))
    (check-equal? (r2 y p ones #:dtype dt) (r2 y p #:dtype dt))
    (check-equal? (mse y p ones #:dtype dt) (mse y p #:dtype dt)))
  (check-equal? (accuracy truth found (make-list 8 1.0)) (accuracy truth found)))

(define (kept xs w)
  (for/list ([x (in-list xs)]
             [c (in-list w)]
             #:unless (zero? c))
    x))

(test-gpu "X6: integer weights match repeated rows, zero weights dropped rows, within 1e-14"
  (for ([c (in-list (sample-cases))])
    (match-define (list i dt y p _) c)
    (define w (integer-weights (length y) (+ 200 i)))
    (define zero-one (map (lambda (x) (if (> x 4) 1.0 0.0)) w))
    (test-case (format "case ~a ~a" i dt)
      (check (lambda (a b) (<= (abs (- a b)) (* 1e-14 (max 1 (abs (- 1 b))))))
             (r2 y p w #:dtype dt)
             (r2 (repeated y w) (repeated p w) #:dtype dt))
      (check (lambda (a b) (<= (abs (- a b)) (* 1e-14 b)))
             (mse y p w #:dtype dt)
             (mse (repeated y w) (repeated p w) #:dtype dt))
      (when (> (length (kept y zero-one)) 1)
        (check (lambda (a b) (<= (abs (- a b)) (* 1e-14 (max 1 (abs (- 1 b))))))
               (r2 y p zero-one #:dtype dt)
               (r2 (kept y zero-one) (kept p zero-one) #:dtype dt))
        (check (lambda (a b) (<= (abs (- a b)) (* 1e-14 b)))
               (mse y p zero-one #:dtype dt)
               (mse (kept y zero-one) (kept p zero-one) #:dtype dt))))))

(test-gpu "X7: mixed float32 and float64 arguments give the bits of the widened call"
  (for ([c (in-list (sample-cases))]
        #:when (eq? (cadr c) 'float32))
    (match-define (list i _ y p w) c)
    (define widened (r2 y p w #:dtype 'float64))
    (define widened-mse (mse y p w #:dtype 'float64))
    (test-case (format "case ~a" i)
      (check-equal? (r2 y p w #:dtype 'float32 #:pred 'float64) widened)
      (check-equal? (r2 y p w #:dtype 'float64 #:pred 'float32) widened)
      (check-equal? (r2 y p w #:dtype 'float32) widened)
      (check-equal? (mse y p w #:dtype 'float32 #:pred 'float64) widened-mse)
      (check-equal? (mse y p w #:dtype 'float64 #:pred 'float32) widened-mse)
      (when w
        (define narrow-w (map flsingle w))
        (check-equal? (r2 y p narrow-w #:dtype 'float32 #:weight 'float32)
                      (r2 y p narrow-w #:dtype 'float64 #:weight 'float64))
        (check-equal? (mse y p narrow-w #:dtype 'float32 #:weight 'float32)
                      (mse y p narrow-w #:dtype 'float64 #:weight 'float64))))))

(test-gpu "accuracy-score takes int32 and int64 in either argument and compares by value"
  (check-equal? (accuracy (list 0 (expt 2 32)) '(0 0) #:true 'int64 #:pred 'int32) 0.5)
  (check-equal? (accuracy '(1 2 3) '(1 2 4)) (exact->inexact 2/3))
  (check-equal? (accuracy '(1 2 3) '(1 2 4) '(1.0 1.0 2.0)) 0.5))
