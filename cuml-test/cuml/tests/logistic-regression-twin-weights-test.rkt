#lang racket/base

(require (only-in racket/list filter-not)
         (only-in racket/match match-define struct*)
         (only-in rackunit check-equal? test-case)
         (only-in "private/linear-data.rkt" dtypes)
         (only-in "private/logistic-data.rkt"
                  blob-case
                  fit-flat
                  fit-parts
                  lr-case
                  lr-case-w
                  lr-case-X
                  lr-case-y
                  penalty-settings
                  setting-fit
                  setting-penalty
                  tight-max-iter
                  tight-tol)
         (only-in "private/logistic-twin-data.rkt" their-fit twin-trial)
         (only-in "private/logistic-twin-rows.rkt" for-each-twin request)
         (only-in "private/python-env.rkt" test-twin))

(define summing-to-n
  (let* ([c (blob-case "binary weighted" 150 4 2 24 #:weighted? #t)]
         [w (lr-case-w c)]
         [scale (/ (length w) (apply + w))])
    (lr-case "binary weights summing to n"
             (lr-case-X c)
             (lr-case-y c)
             (map (lambda (v) (* v scale)) w))))

(define penalised (filter-not (lambda (s) (eq? (setting-penalty s) 'none)) penalty-settings))

(define grid
  (for*/list ([dt (in-list dtypes)]
              [s (in-list penalised)])
    (twin-trial dt summing-to-n s #t)))

(define (tight-request r)
  (match-define (struct* twin-trial ([dt dt])) r)
  (request r #:op "cuml-fit" #:tol (tight-tol dt) #:max-iter tight-max-iter))

(test-twin "weights summing to n fit as Python cuML's bit for bit"
  (for-each-twin
   (lambda (r theirs)
     (match-define (struct* twin-trial ([dt dt] [setting s])) r)
     (test-case (format "~a ~a" dt (setting-penalty s))
       (define ours
         (fit-parts (setting-fit summing-to-n dt s #:tol (tight-tol dt) #:max-iter tight-max-iter)))
       (check-equal? (fit-flat ours) (fit-flat (their-fit theirs dt)))))
   grid
   #:request tight-request))
