#lang racket/base

(require (only-in cuml/metrics mean-squared-error r2-score)
         (only-in racket/list take)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in "private/metrics-data.rkt" regression-cases regression-pair)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/scaler-data.rkt" rows-of)
         (only-in "private/scores-data.rkt" on weights-of)
         (only-in "private/scores-twin-data.rkt" case-of checked-twin-results))

(define (vector-cases)
  (for/list ([c (in-list (regression-cases))])
    (match-define (list i dt y p w) c)
    (list (format "case ~a ~a" i dt) dt y p w)))

(define (matrix-cases)
  (for/list ([i (in-range 10)])
    (define dt (if (even? i) 'float32 'float64))
    (define d (+ 2 (modulo i 4)))
    (define n 200)
    (define-values (y p w)
      (regression-pair (* n d)
                       #:offset (* 100.0 i)
                       #:noise 0.5
                       #:seed (+ 9600 i)
                       #:dtype dt
                       #:weighted? (odd? (quotient i 2))))
    (list (format "matrix ~a ~a ~ax~a" i dt n d) dt (rows-of y d) (rows-of p d) (and w (take w n)))))

(define (relative ours theirs)
  (/ (abs (- ours theirs)) (abs theirs)))

(define (r2-relative ours theirs)
  (/ (abs (- ours theirs)) (max 1.0 (abs (- 1.0 theirs)))))

(test-twin "T1, T2: r2 and the error agree with sklearn on the float64 cast and Python cuML on the dtype"
  (define vectors (vector-cases))
  (define matrices (matrix-cases))
  (define r2-results
    (checked-twin-results (for/list ([c (in-list vectors)])
                            (match-define (list _ dt y p w) c)
                            (case-of "r2" y p w #:dtype dt))))
  (define mse-results
    (checked-twin-results (for/list ([c (in-list (append vectors matrices))])
                            (match-define (list _ dt y p w) c)
                            (case-of "mse" y p w #:dtype dt))))
  (define (check-against score cases results gap cuml-bound)
    (for/fold ([worst (hash)])
              ([c (in-list cases)]
               [theirs (in-list results)])
      (match-define (list name dt y p w) c)
      (match-define (hash-table ('sklearn sk) ('cuml cu)) theirs)
      (define ours (score (on y dt) (on p dt) #:sample-weight (weights-of w)))
      (define sk-gap (gap ours sk))
      (define cu-gap (gap ours cu))
      (test-case name
        (check-true (<= sk-gap 1e-14) (format "T1: ~a against sklearn's ~a" ours sk))
        (check-true (<= cu-gap (if (eq? dt 'float32) cuml-bound 1e-14))
                    (format "T2: ~a against Python cuML's ~a" ours cu)))
      (hash-update (hash-update worst (cons 'sklearn dt) (lambda (g) (max g sk-gap)) 0.0)
                   (cons 'cuml dt)
                   (lambda (g) (max g cu-gap))
                   0.0)))
  (printf "twin r2: worst gaps ~a\n" (check-against r2-score vectors r2-results r2-relative 1e-6))
  (printf "twin error: worst gaps ~a\n"
          (check-against mean-squared-error (append vectors matrices) mse-results relative 3e-7)))
