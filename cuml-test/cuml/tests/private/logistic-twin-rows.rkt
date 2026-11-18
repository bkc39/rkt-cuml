#lang racket/base

(require (only-in racket/list remove-duplicates)
         (only-in racket/match match-define)
         (only-in "linear-data.rkt" dtypes)
         (only-in "logistic-data.rkt"
                  blob-case
                  default-tol
                  lr-case-name
                  lr-case-y
                  penalty-settings
                  setting-fit
                  setting-penalty)
         (only-in "logistic-twin-data.rkt" setting-twin twin-trial))

(provide classes-of
         repeatable?
         request
         row-fit
         row-name
         rows)

(define cases
  (list (blob-case "binary" 150 4 2 21)
        (blob-case "three classes" 150 4 3 22)
        (blob-case "five classes" 200 4 5 23)))

(define rows
  (for*/list ([dt (in-list dtypes)]
              [c (in-list cases)]
              [s (in-list penalty-settings)]
              [fi (in-list '(#t #f))])
    (twin-trial dt c s fi)))

(define (request r #:op [op "cuml-apply"] #:tol [tol default-tol] #:max-iter [max-iter 1000])
  (match-define (twin-trial dt c s fi) r)
  (setting-twin op c dt s #:fit-intercept fi #:tol tol #:max-iter max-iter))

(define (classes-of c)
  (length (remove-duplicates (lr-case-y c))))

(define (repeatable? r)
  (match-define (twin-trial dt c s _) r)
  (not (and (eq? dt 'float32) (> (classes-of c) 2) (memq (setting-penalty s) '(l1 elasticnet)))))

(define (row-fit r)
  (match-define (twin-trial dt c s fi) r)
  (setting-fit c dt s #:fit-intercept fi))

(define (row-name r)
  (match-define (twin-trial dt c s fi) r)
  (format "~a ~a ~a ~a" dt (lr-case-name c) (setting-penalty s) fi))
