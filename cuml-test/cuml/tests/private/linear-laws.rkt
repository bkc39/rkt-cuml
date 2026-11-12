#lang racket/base

(require (only-in cuml/linear-model linear-regression-predict)
         (only-in racket/list count)
         (only-in racket/match match-define)
         (only-in rackunit test-case)
         (only-in raft device-vector->list)
         (only-in threading ~>)
         (only-in "exact-algebra.rkt" dot norm2)
         (only-in "linear-data.rkt"
                  algorithms
                  case-fit
                  case-result
                  check-law
                  check-law-case
                  columns-of
                  dtypes
                  exact-fit
                  gap
                  host-cosine
                  lcase-X
                  lcase-name
                  lcase-w
                  lcase-y
                  max-abs
                  on-matrix
                  oracle-lstsq
                  result-fit
                  solution
                  staging
                  staging-of
                  unit-roundoff
                  weighted-spread
                  weights-or-ones))

(provide check-lstsq-fit
         check-oracle-laws
         check-prediction-law)

(define thick-rows (expt 2 17))

(define (sum-roundings rows)
  (if (< rows thick-rows) 15 21))

(define (mean-roundings rows weighted?)
  (define h (sum-roundings rows))
  (if weighted?
      (+ (* 2 h) 2)
      (+ h 2)))

(define (gamma k u)
  (/ (* k u) (- 1 (* k u))))

(define (check-lstsq-fit c fi u fit exact staged)
  (match-define (solution c* b* _) exact)
  (match-define (staging scale y-scale weight-scale kept kappa bottom) staged)
  (match-define (list coef b) (exact-fit fit))
  (define X (lcase-X c))
  (define y (lcase-y c))
  (define weights (weights-or-ones (lcase-w c) y))
  (define total (apply + weights))
  (define columns (columns-of X))
  (define mean-gamma (gamma (mean-roundings (count positive? weights) (lcase-w c)) u))
  (define (abs-sum xs)
    (dot weights (map abs xs)))
  (define (mean-error xs)
    (if fi
        (/ (* mean-gamma (abs-sum xs)) total)
        0))
  (define errors (map mean-error columns))
  (define y-error (mean-error y))
  (define (shift cs)
    (+ y-error
       (for/sum ([e (in-list errors)] [cj (in-list cs)] [k (in-list kept)] #:when k) (* e (abs cj)))))
  (define (staged-coef cs)
    (map (lambda (cj dj) (/ (* y-scale cj) dj)) cs scale))
  (define staged-errors
    (for/list ([e (in-list errors)] [dj (in-list scale)] [k (in-list kept)] #:when k)
      (* dj e)))
  (define centring-coef
    (cond
      [(not fi) 0]
      [(zero? bottom) +inf.0]
      [else (/ (* y-scale (norm2 staged-errors) weight-scale total (shift c*)) bottom)]))
  (define coef-bound (+ (* 128 u kappa kappa (max (max-abs (staged-coef c*)) 1)) centring-coef))
  (check-law-case "T1" (gap (staged-coef coef) (staged-coef c*)) coef-bound)
  (define y-square (and fi (weighted-spread y weights)))
  (define-values (centring-cosine column-stretch)
    (for/fold ([worst 0] [stretch 0])
              ([column (in-list columns)] [e (in-list errors)] [k (in-list kept)] #:when (and fi k))
      (define square (weighted-spread column weights))
      (values (max worst (/ (* total e (shift coef)) (sqrt (* square y-square))))
              (max stretch (/ (* total e e) square)))))
  (define stretch
    (if fi
        (sqrt (* (+ 1 column-stretch) (+ 1 (/ (* total y-error y-error) y-square))))
        1))
  (check-law-case "T3"
                  (host-cosine X y (lcase-w c) coef b fi)
                  (+ (* 64 u kappa stretch) centring-cosine))
  (when fi
    (define sum-bound
      (+ (* u total (abs b))
         (* mean-gamma (abs-sum y))
         (* (+ mean-gamma (gamma (length coef) u))
            (for/sum ([column (in-list columns)] [cj (in-list coef)]) (* (abs-sum column) (abs cj))))))
    (define staged-means
      (for/sum ([column (in-list columns)] [dj (in-list scale)] [k (in-list kept)] #:when k)
               (* (abs (/ (dot weights column) total)) dj)))
    (check-law-case "T2" (abs (- b b*)) (+ (/ sum-bound total) (/ (* staged-means coef-bound) y-scale)))
    (define residual-sum
      (for/sum ([row (in-list X)] [yi (in-list y)] [wi (in-list weights)])
               (* wi (- yi b (dot row coef)))))
    (check-law-case "T4" (abs residual-sum) sum-bound)))

(define (check-oracle-laws cases intercepts)
  (for* ([c (in-list cases)]
         [fi (in-list intercepts)])
    (define exact (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) fi))
    (define staged (staging-of (lcase-X c) (lcase-y c) (lcase-w c) fi))
    (for* ([dt (in-list dtypes)]
           [a (in-list algorithms)])
      (test-case (format "~a ~a ~a ~a" (lcase-name c) fi dt a)
        (check-lstsq-fit c
                         fi
                         (unit-roundoff dt)
                         (case-fit c dt #:algorithm a #:fit-intercept fi)
                         exact
                         staged)))))

(define (check-prediction-law cases)
  (for* ([c (in-list cases)]
         [dt (in-list dtypes)]
         #:unless (lcase-w c))
    (test-case (format "~a ~a" (lcase-name c) dt)
      (define u (unit-roundoff dt))
      (define r (case-result c dt))
      (match-define (list coef b) (exact-fit (result-fit r)))
      (define d (length coef))
      (define p (~> r (linear-regression-predict (on-matrix (lcase-X c) dt)) device-vector->list))
      (for ([row (in-list (lcase-X c))]
            [pi (in-list p)])
        (define exact (+ b (for/sum ([x (in-list row)] [cj (in-list coef)]) (* x cj))))
        (define size (+ (abs b) (for/sum ([x (in-list row)] [cj (in-list coef)]) (abs (* x cj)))))
        (check-law "T12" (abs (- (inexact->exact pi) exact)) (* 2 d u size))))))

(module+ test
  (require (only-in rackunit check-equal?))
  (check-equal? (map (lambda (rows) (mean-roundings rows #f)) (list (- thick-rows 1) thick-rows))
                '(17 23))
  (check-equal? (map (lambda (rows) (mean-roundings rows #t)) (list (- thick-rows 1) thick-rows))
                '(32 44)))
