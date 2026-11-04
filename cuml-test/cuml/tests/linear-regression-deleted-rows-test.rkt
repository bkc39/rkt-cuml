#lang racket/base

(require (only-in racket/list make-list take)
         (only-in rackunit check-false check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" uniform-source)
         (only-in "private/linear-data.rkt" algorithms check-law dtypes fit-gap fit-of unit-roundoff)
         (only-in "private/scaler-data.rkt" bits-equal?))

(define (kept xs gone)
  (for/list ([x (in-list xs)]
             [i (in-naturals)]
             #:unless (memv i gone))
    x))

(test-gpu "with every column constant, a zero or cut-off weight is a deleted row, bitwise"
  (for* ([n (in-list '(100 1000 5000 20000))]
         [seed (in-range 4)]
         [cut? (in-list '(#f #t))])
    (test-case (format "n ~a seed ~a~a" n seed (if cut? " cut off" ""))
      (define draw (uniform-source (+ 17 seed n)))
      (define y
        (for/list ([_ (in-range n)])
          (* 1000.0 (draw))))
      (define scale (if cut? 1e300 1.0))
      (define w
        (for/list ([_ (in-range n)])
          (* scale (+ 0.1 (draw)))))
      (define gone
        (for/list ([k (in-range 7)])
          (* k (quotient n 7))))
      (define zeroed
        (for/list ([v (in-list w)]
                   [i (in-naturals)])
          (cond
            [(not (memv i gone)) v]
            [cut? 1e-30]
            [else 0.0])))
      (define X (make-list n '(3.0)))
      (check-true (bits-equal? (fit-of X y zeroed 'float64)
                               (fit-of (kept X gone) (kept y gone) (kept w gone) 'float64))))))

(define cutoff-X '((0 1) (1 1) (2 1) (3 5)))
(define cutoff-y '(1 2 3 100))

(define (cutoff-of dt)
  (if (eq? dt 'float32)
      (values (expt 2 -148) (expt 2 -149))
      (values (expt 2 -1073) (expt 2 -1074))))

(test-gpu "a weight at the cutoff deletes its row and one step above it counts"
  (for* ([dt (in-list dtypes)]
         [a (in-list algorithms)])
    (test-case (format "~a ~a" dt a)
      (define-values (cutoff step) (cutoff-of dt))
      (define deleted (fit-of (take cutoff-X 3) (take cutoff-y 3) '(3 3 3) dt #:algorithm a))
      (check-true (bits-equal? (fit-of cutoff-X cutoff-y (list 3 3 3 cutoff) dt #:algorithm a)
                               deleted)
                  "at the cutoff")
      (define counted (fit-of cutoff-X cutoff-y (list 3 3 3 (+ cutoff step)) dt #:algorithm a))
      (check-false (bits-equal? counted deleted) "one step above")
      (unless (eq? a 'auto)
        (check-law "one step above, the second column fits the last row"
                   (fit-gap counted '((1 24) -23))
                   (* 64 (unit-roundoff dt) 24))))))
