#lang racket/base

(require (only-in racket/list filter-not)
         (only-in racket/match match-define)
         (only-in rackunit check-true test-case)
         (only-in "private/linear-data.rkt" by-dtype dtypes)
         (only-in "private/logistic-data.rkt"
                  blob-case
                  fit-flat
                  fit-parts
                  lr-case
                  lr-case-w
                  lr-case-X
                  lr-case-y
                  penalty-settings
                  relative-gap
                  setting-fit
                  setting-penalty
                  tight-max-iter
                  tight-tol)
         (only-in "private/logistic-twin-data.rkt" logistic-twin-results their-fit twin-trial)
         (only-in "private/logistic-twin-rows.rkt" request)
         (only-in "private/python-env.rkt" test-twin))

(define summing-to-n
  (let* ([c (blob-case "binary weighted" 150 4 2 24 #:weighted? #t)]
         [w (lr-case-w c)]
         [scale (/ (length w) (apply + w))])
    (lr-case "binary weights summing to n"
             (lr-case-X c)
             (lr-case-y c)
             (map (lambda (v) (* v scale)) w))))

(define (weighted-gap-bound dt)
  (by-dtype dt 0 (min (* 64 1e-6) 3e-2)))

(define penalised (filter-not (lambda (s) (eq? (setting-penalty s) 'none)) penalty-settings))

(test-twin "weights summing to n fit as Python cuML's in float32 and near its optimum in float64"
  (define grid
    (for*/list ([dt (in-list dtypes)]
                [s (in-list penalised)])
      (twin-trial dt summing-to-n s #t)))
  (define (tight-request g)
    (match-define (twin-trial dt _ _ _) g)
    (request g #:op "cuml-fit" #:tol (tight-tol dt) #:max-iter tight-max-iter))
  (for ([g (in-list grid)]
        [theirs (in-list (logistic-twin-results (map tight-request grid)))])
    (match-define (twin-trial dt _ s _) g)
    (test-case (format "~a ~a" dt (setting-penalty s))
      (define ours
        (fit-parts (setting-fit summing-to-n dt s #:tol (tight-tol dt) #:max-iter tight-max-iter)))
      (define gap (relative-gap (fit-flat ours) (fit-flat (their-fit theirs dt))))
      (check-true (<= gap (weighted-gap-bound dt)) (format "gap ~a" gap)))))
