#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-true test-case)
         (only-in "private/linear-data.rkt" by-dtype dtypes)
         (only-in "private/mbsgd-data.rkt" blob-rows normal-rows reg-target)
         (only-in "private/mbsgd-twin-data.rkt"
                  clf-fit
                  mbsgd-twin-results
                  overridden
                  reg-fit
                  twin-request)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/sgd-replay.rkt" sgd-objective))

(define rows (normal-rows 256 4 5))
(define targets (reg-target rows 5 #:noise 0.5))
(define-values (blobs labels) (blob-rows 256 3 4 #:sep 1.0))

(define full-batch (hash '#:batch-size 256 '#:shuffle #f '#:tol 0.0 '#:eta0 0.05 '#:epochs 1000))

(define arms
  (list (list 'squared-loss (hash '#:penalty 'l2 '#:alpha 0.1))
        (list 'squared-loss (hash '#:penalty 'l1 '#:alpha 0.05))
        (list 'squared-loss (hash '#:penalty 'elasticnet '#:alpha 0.1 '#:l1-ratio 0.15))
        (list 'squared-loss (hash '#:penalty 'none))
        (list 'log (hash '#:alpha 1e-2 '#:eta0 0.5 '#:fit-intercept #f))))

(define (settled more)
  (overridden full-batch more))

(define (objective loss settings X y w b)
  (sgd-objective loss
                 (hash-ref settings '#:penalty 'l2)
                 (hash-ref settings '#:alpha 1e-4)
                 (hash-ref settings '#:l1-ratio 0.15)
                 X
                 y
                 w
                 b))

(define (problem loss)
  (if (eq? loss 'log)
      (values blobs labels)
      (values rows targets)))

(define (ours dt loss settings)
  (if (eq? loss 'log)
      (clf-fit dt blobs labels 'log settings)
      (reg-fit dt rows targets settings)))

(test-twin "the iterate's objective is sklearn's optimum objective"
  (define requests
    (for/list ([arm (in-list arms)])
      (match-define (list loss more) arm)
      (define-values (X y) (problem loss))
      (if (eq? loss 'log)
          (twin-request "sklearn-opt" 'float64 X (settled more) #:codes y #:loss loss)
          (twin-request "sklearn-opt" 'float64 X (settled more) #:y y #:loss loss))))
  (for ([arm (in-list arms)]
        [theirs (in-list (mbsgd-twin-results requests))])
    (match-define (list loss more) arm)
    (define settings (settled more))
    (define-values (X y) (problem loss))
    (define (objective-at w b)
      (objective loss settings X y w b))
    (define best (objective-at (hash-ref theirs 'coef) (hash-ref theirs 'intercept)))
    (test-case (format "~a ~a" loss more)
      (check-true (<= (abs (- best (hash-ref theirs 'objective))) (* 1e-13 best))
                  (format "objective here ~a, sklearn's ~a" best (hash-ref theirs 'objective)))
      (for ([dt (in-list dtypes)])
        (match-define (list w b) (ours dt loss settings))
        (define excess (/ (- (objective-at w b) best) best))
        (check-true (<= excess (by-dtype dt 1e-6 1e-12)) (format "~a excess ~a" dt excess))))))
