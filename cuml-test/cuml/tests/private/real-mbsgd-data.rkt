#lang racket/base

(require (only-in cuml/linear-model
                  mbsgd-classifier-fit
                  mbsgd-classifier-predict
                  mbsgd-classifier-result-classes)
         (only-in racket/match match-define struct*)
         (only-in rackunit check-equal? check-true test-case)
         (only-in "elastic-net-data.rkt" apply-keywords)
         (only-in "linear-data.rkt" on-matrix oracle-ridge ridge-solution)
         (only-in "logistic-data.rkt" correct lr-case-X)
         (only-in "mbsgd-twin-data.rkt" mbsgd-twin-results reg-fit their-parts twin-request)
         (only-in "real-linear-data.rkt" set-data)
         (only-in "real-logistic-data.rkt" labelled labelled-set standardised)
         (only-in "sgd-replay.rkt" sgd-objective))

(provide check-breast-cancer
         check-diabetes-gap
         check-diabetes-twin
         diabetes)

(define ridge-alpha 1e-4)

(define (diabetes #:standardise? [standardise? #f])
  (match-define (cons X y) (set-data 'diabetes))
  (values (for/list ([r (in-list (if standardise?
                                     (standardised X)
                                     X))])
            (map exact->inexact r))
          (map exact->inexact y)))

(define (objective X y w b)
  (sgd-objective 'squared-loss 'l2 ridge-alpha 0 X y w b))

(define (optimum X y)
  (match-define (struct* ridge-solution ([coef coef] [intercept intercept]))
    (oracle-ridge (for/list ([r (in-list X)])
                    (map inexact->exact r))
                  (map inexact->exact y)
                  #f
                  (* (length y) (inexact->exact ridge-alpha))
                  #t))
  (objective X y (map exact->inexact coef) (exact->inexact intercept)))

(define (check-diabetes-gap X y dt settings low high #:random-state [seed #f])
  (define best (optimum X y))
  (match-define (list w b) (reg-fit dt X y settings #:random-state seed))
  (define gap (/ (- (objective X y w b) best) best))
  (check-true (<= low gap high)
              (format "random-state ~a: gap (F - F*)/F* ~a, expected in [~a, ~a]" seed gap low high)))

(define (check-diabetes-twin X y dt)
  (match-define (list theirs) (mbsgd-twin-results (list (twin-request "cuml-reg" dt X (hash) #:y y))))
  (check-equal? (reg-fit dt X y (hash)) (their-parts theirs dt)))

(define (accuracy predicted labels)
  (/ (correct predicted labels) (length labels)))

(define (check-breast-cancer dt)
  (match-define (struct* labelled ((case c) [labels labels] [classes classes]))
    (labelled-set 'breast-cancer "diagnosis" '("id") #:standardise? #t))
  (define X (on-matrix (lr-case-X c) dt))
  (for ([arm (in-list (list (list "hinge defaults" (hash '#:loss 'hinge) 0.975)
                            (list "log defaults" (hash '#:loss 'log) 0.955)
                            (list "log without an intercept, eta0 1e-3"
                                  (hash '#:loss 'log '#:fit-intercept #f '#:eta0 1e-3)
                                  0.95)))])
    (match-define (list name settings bound) arm)
    (test-case (format "~a ~a" dt name)
      (define fitted (apply-keywords mbsgd-classifier-fit settings (list X labels)))
      (check-equal? (mbsgd-classifier-result-classes fitted) classes "classes")
      (define correct (accuracy (mbsgd-classifier-predict fitted X) labels))
      (check-true (>= correct bound)
                  (format "train accuracy ~a, bound ~a" (exact->inexact correct) bound)))))
