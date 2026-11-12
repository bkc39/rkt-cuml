#lang racket/base

(require (only-in cuml/linear-model linear-regression-predict)
         (only-in racket/list count)
         (only-in racket/match match-define)
         (only-in rackunit test-case)
         (only-in raft device-vector->list)
         (only-in threading ~>)
         (only-in "exact-algebra.rkt" bracket-condition dot norm2 outer-bracket)
         (only-in "linear-data.rkt"
                  algorithms
                  case-fit
                  case-result
                  check-law
                  check-law-case
                  columns-of
                  dtypes
                  exact-fit
                  floor-log2
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
                  unit-roundoff
                  weighted-gram
                  weighted-spread
                  weights-or-ones))

(provide (struct-out staging)
         check-lstsq-fit
         check-oracle-laws
         check-prediction-law
         staging-of)

(struct staging (scale y-scale weight-scale kept kappa bottom) #:transparent)

(define (even-floor k)
  (- k (modulo k 2)))

(define (weight-scale w)
  (if w
      (expt 2 (~> w max-abs floor-log2 - even-floor))
      1))

(define (staging-of X y w fit-intercept)
  (define p (weight-scale w))
  (define-values (G _b _mx _my)
    (weighted-gram X y (and w (map (lambda (v) (* p v)) w)) fit-intercept))
  (define weights (weights-or-ones w y))
  (define exponents
    (for/list ([row (in-list G)]
               [j (in-naturals)])
      (define g (list-ref row j))
      (and (positive? g) (~> g floor-log2 (/ 2) floor -))))
  (define kept (filter values exponents))
  (define rows (count positive? weights))
  (define shared (and (> (length kept) (- rows (if fit-intercept 1 0))) (apply min kept)))
  (define scale
    (for/list ([e (in-list exponents)])
      (expt 2 (or shared e 0))))
  (define staged
    (for/list ([row (in-list G)]
               [di (in-list scale)]
               [ei (in-list exponents)]
               #:when ei)
      (for/list ([g (in-list row)]
                 [dj (in-list scale)]
                 [ej (in-list exponents)]
                 #:when ej)
        (* di g dj))))
  (define largest
    (for/fold ([m 0])
              ([yi (in-list y)]
               [wi (in-list weights)]
               #:when (positive? wi))
      (max m (abs yi))))
  (define y-scale
    (cond
      [(zero? largest) 1]
      [else (expt 2 (- (floor-log2 largest)))]))
  (define outer
    (cond
      [(null? staged) '(1 . 1)]
      [else (outer-bracket staged)]))
  (staging scale
           y-scale
           p
           (map (lambda (e) (and e #t)) exponents)
           (bracket-condition outer)
           (cdr outer)))

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
    (for/list ([e (in-list errors)]
               [dj (in-list scale)]
               [k (in-list kept)]
               #:when k)
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
    (for/fold ([worst 0]
               [stretch 0])
              ([column (in-list columns)]
               [e (in-list errors)]
               [k (in-list kept)]
               #:when (and fi k))
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
            (for/sum ([column (in-list columns)] [cj (in-list coef)])
                     (* (abs-sum column) (abs cj))))))
    (define staged-means
      (for/sum ([column (in-list columns)] [dj (in-list scale)] [k (in-list kept)] #:when k)
               (* (abs (/ (dot weights column) total)) dj)))
    (check-law-case "T2"
                    (abs (- b b*))
                    (+ (/ sum-bound total) (/ (* staged-means coef-bound) y-scale)))
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
  (require (only-in rackunit check-equal? check-true)
           (only-in "exact-algebra.rkt" condition))
  (define two-column-gram
    (list (list 5/4 (* 2500 (expt 2 -12))) (list (* 2500 (expt 2 -12)) (* 8750000 (expt 2 -22)))))
  (check-equal? (staging-of '((1 1000) (2 3000) (4 2000) (3 5000)) '(1 3 2 4) #f #t)
                (staging (list 1/2 (expt 2 -11))
                         1/4
                         1
                         '(#t #t)
                         (condition two-column-gram)
                         (cdr (outer-bracket two-column-gram))))
  (match-define (staging weighted-scale weighted-y-scale weighted-power _ weighted-kappa _)
    (staging-of '((1 1000) (2 3000) (4 2000) (3 5000)) '(1 3 2 4) '(1 4 4 1) #t))
  (check-equal? (list weighted-scale weighted-y-scale weighted-power)
                (list (list 1 (expt 2 -10)) 1/4 1/4))
  (check-equal? weighted-kappa
                (condition (list (list 29/10 (* -200 (expt 2 -10)))
                                 (list (* -200 (expt 2 -10)) (* 2600000 (expt 2 -20))))))
  (match-define (staging wide-scale _ _ _ _ _) (staging-of '((1 64) (2 0)) '(1 2) #f #t))
  (check-equal? wide-scale (list 1/32 1/32))
  (match-define (staging constant-scale _ _ constant-kept constant-kappa _)
    (staging-of '((1 5) (2 5) (4 5)) '(1 2 3) #f #t))
  (check-equal? constant-scale (list 1/2 1))
  (check-equal? constant-kept '(#t #f))
  (check-true (<= 1 constant-kappa 2) (format "κ over the varying column ~a" constant-kappa))
  (check-equal? (map (lambda (rows) (mean-roundings rows #f)) (list (- thick-rows 1) thick-rows))
                '(17 23))
  (check-equal? (map (lambda (rows) (mean-roundings rows #t)) (list (- thick-rows 1) thick-rows))
                '(32 44)))
