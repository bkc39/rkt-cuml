#lang racket/base

(require (only-in cuml/linear-model
                  mbsgd-classifier-fit
                  mbsgd-classifier-predict
                  mbsgd-classifier-result-classes)
         (only-in datasets/core dataset-table-columns dataset-table-names load-dataset-table)
         (only-in racket/match match-define struct*)
         (only-in racket/math sqr)
         (only-in rackunit check-equal? check-true test-case)
         (only-in threading ~>)
         (only-in "elastic-net-data.rkt" apply-keywords)
         (only-in "linear-data.rkt" columns-of on-matrix oracle-ridge ridge-solution)
         (only-in "logistic-data.rkt" lr-case-X)
         (only-in "mbsgd-twin-data.rkt" mbsgd-twin-results reg-fit their-parts twin-request)
         (only-in "real-logistic-data.rkt" labelled labelled-set))

(provide check-breast-cancer
         check-diabetes-gap
         check-diabetes-twin
         diabetes)

(define ridge-alpha 1e-4)

(define (standardised X)
  (define (scaled xs)
    (define n (length xs))
    (define m (/ (apply + xs) n))
    (define s (sqrt (/ (for/sum ([x (in-list xs)]) (sqr (- x m))) n)))
    (map (lambda (x) (/ (- x m) s)) xs))
  (~> X columns-of (map scaled _) columns-of))

(define (diabetes #:standardise? [standardise? #f])
  (define table (load-dataset-table 'diabetes))
  (define named
    (for/hash ([n (in-list (dataset-table-names table))]
               [column (in-vector (dataset-table-columns table))])
      (values n (map real->double-flonum (vector->list column)))))
  (define predictors (remove "response" (dataset-table-names table)))
  (define X
    (columns-of (for/list ([p (in-list predictors)])
                  (hash-ref named p))))
  (values (if standardise?
              (standardised X)
              X)
          (hash-ref named "response")))

(define (objective X y w b)
  (+ (/ (for/sum ([x (in-list X)] [v (in-list y)])
                 (sqr (- v b (for/sum ([xi (in-list x)] [wi (in-list w)]) (* xi wi)))))
        (length y))
     (* ridge-alpha (for/sum ([wi (in-list w)]) (sqr wi)))))

(define (optimum X y)
  (match-define (struct* ridge-solution ([coef coef] [intercept intercept]))
    (oracle-ridge (for/list ([r (in-list X)])
                    (map inexact->exact r))
                  (map inexact->exact y)
                  #f
                  (* (length y) (inexact->exact ridge-alpha))
                  #t))
  (objective X y (map exact->inexact coef) (exact->inexact intercept)))

(define (check-diabetes-gap X y dt settings low high)
  (define best (optimum X y))
  (match-define (list w b) (reg-fit dt X y settings))
  (define gap (/ (- (objective X y w b) best) best))
  (check-true (<= low gap high) (format "gap (F - F*)/F* ~a, expected in [~a, ~a]" gap low high)))

(define (check-diabetes-twin X y dt)
  (match-define (list theirs) (mbsgd-twin-results (list (twin-request "cuml-reg" dt X (hash) #:y y))))
  (check-equal? (reg-fit dt X y (hash)) (their-parts theirs dt)))

(define (accuracy predicted labels)
  (/ (for/sum ([p (in-list predicted)] [l (in-list labels)]) (if (equal? p l) 1 0)) (length labels)))

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
