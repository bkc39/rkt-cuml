#lang racket/base

(require (only-in cuml elastic-net fit lasso linear-regression predict ridge score)
         (only-in cuml/linear-model lasso-fit)
         (only-in datasets/core dataset-table-columns dataset-table-names load-dataset-table)
         (only-in racket/list make-list)
         (only-in racket/match match match-define struct*)
         (only-in racket/math sqr)
         (only-in rackunit check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "cuml-error.rkt" check-cuml-error)
         (only-in "elastic-net-data.rkt"
                  apply-keywords
                  check-distance-laws
                  check-kkt-laws
                  default-max-iter
                  default-tol
                  en-case-fit
                  en-run
                  rows-of)
         (only-in "elastic-net-oracle.rkt"
                  alpha-max
                  en-problem
                  en-problem-G
                  en-problem-total
                  en-solution-coef
                  oracle-elastic-net
                  problem-of)
         (only-in "exact-algebra.rkt"
                  bracket-condition
                  dot
                  eigen-bracket
                  min-norm-solve
                  varying-inner)
         (only-in "linear-data.rkt"
                  case-fit
                  columns-of
                  lcase
                  lcase-name
                  lcase-w
                  lcase-X
                  lcase-y
                  max-abs
                  on-matrix
                  on-vector
                  ridge-solve
                  rounded-case
                  solution
                  unit-roundoff)
         (only-in "linear-laws.rkt" check-lstsq-fit)
         (only-in "metrics-data.rkt" exact-r2 r2-bound r2-gap)
         (only-in "ridge-data.rkt" check-ridge-fit ridge-case-fit solvers))

(provide test-real-set)

(struct real-case
        (case dtype
          problem
          inner
          outer)
  #:transparent)
(struct real-set (response excluded))

(define real-sets
  (hasheq 'diabetes (real-set "response" '())
          'mtcars (real-set "mpg" '("model"))
          'wine (real-set "alcohol" '("cultivar"))
          'breast-cancer (real-set "texture-mean" '("diagnosis" "id"))))

(define all-arms
  '((raw . unweighted) (raw . weighted) (standardised . unweighted) (standardised . weighted)))
(define all-dtypes '(float32 float64))

(define lstsq-algorithms '(auto eig svd svd-jacobi qr svd-qr))
(define l1-ratios '(1 0.5 0.1))
(define path-steps '(1 2 3 4 5))
(define above-alpha-max (+ 1 1e-6))
(define ridge-path-l1-ratio 1/1000)
(define tight-max-iter 100000)
(define weight-seed 2026)

(define (tight-tol dt)
  (if (eq? dt 'float32) 1e-7 1e-10))

(define (flonum-exact v)
  (inexact->exact (real->double-flonum v)))

(define (set-data name)
  (match-define (real-set response excluded) (hash-ref real-sets name))
  (define table (load-dataset-table name))
  (define named
    (for/hash ([n (in-list (dataset-table-names table))]
               [column (in-vector (dataset-table-columns table))])
      (values n column)))
  (define (values-of n)
    (map flonum-exact (vector->list (hash-ref named n))))
  (define predictors (remove* (cons response excluded) (dataset-table-names table)))
  (values (columns-of (map values-of predictors)) (values-of response)))

(define (seeded-weights n)
  (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
    (random-seed weight-seed)
    (for/list ([_ (in-range n)])
      (+ 1 (random 4)))))

(define (standardised X weights)
  (define total (apply + weights))
  (define (scaled xs)
    (define m (/ (dot weights xs) total))
    (define s
      (~> (for/sum ([x (in-list xs)] [w (in-list weights)]) (* w (sqr (- x m))))
          (/ total)
          exact->inexact
          sqrt
          inexact->exact))
    (map (lambda (x) (flonum-exact (/ (- x m) s))) xs))
  (~> X columns-of (map scaled _) columns-of))

(define made-cases (make-hash))

(define (real-cases name dts arms)
  (hash-ref! made-cases (list name dts arms) (lambda () (made-real-cases name dts arms))))

(define (made-real-cases name dts arms)
  (define-values (X y) (set-data name))
  (for*/list ([arm (in-list arms)]
              [dt (in-list dts)])
    (match-define (cons scaling weighting) arm)
    (define w (and (eq? weighting 'weighted) (seeded-weights (length y))))
    (define rows
      (match scaling
        ['raw X]
        ['standardised (standardised X (or w (make-list (length y) 1)))]))
    (define c (rounded-case (lcase (format "~a ~a ~a ~a" name scaling weighting dt) rows y w) dt))
    (define p (problem-of (lcase-X c) (lcase-y c) (lcase-w c) #t))
    (define G (en-problem-G p))
    (define-values (inner outer) (eigen-bracket G))
    (real-case c dt p (varying-inner G inner) outer)))

(define (lstsq-solution rc)
  (match-define (struct* real-case ([problem p] [outer outer])) rc)
  (match-define (struct* en-problem ([G G] [b b] [mx mx] [my my])) p)
  (define coef (min-norm-solve G b))
  (solution coef (- my (dot mx coef)) (bracket-condition outer)))

(define (ridge-lambda-max rc)
  (match-define (struct* real-case ([problem p])) rc)
  (match-define (struct* en-problem ([total total] [yy yy])) p)
  (exact->inexact (/ (alpha-max p ridge-path-l1-ratio) (sqrt (exact->inexact (/ yy total))))))

(define (path-alpha top k)
  (* top (expt 10.0 (* -3/5 k))))

(define (ridge-alpha rc k)
  (* (exact->inexact (en-problem-total (real-case-problem rc))) (path-alpha (ridge-lambda-max rc) k)))

(define (ridge-oracle rc alpha)
  (match-define (struct* real-case ([problem p] [inner inner] [outer outer])) rc)
  (match-define (struct* en-problem ([G G] [b b] [mx mx] [my my])) p)
  (ridge-solve G b mx my inner outer (inexact->exact alpha)))

(define (check-real-lstsq rc)
  (match-define (struct* real-case ((case c) [dtype dt])) rc)
  (define o (lstsq-solution rc))
  (for ([a (in-list lstsq-algorithms)])
    (test-case (format "~a ~a" (lcase-name c) a)
      (check-lstsq-fit c #t (unit-roundoff dt) (case-fit c dt #:algorithm a) o))))

(define (check-real-ridge rc)
  (match-define (struct* real-case ((case c) [dtype dt])) rc)
  (for ([k (in-list path-steps)])
    (define alpha (ridge-alpha rc k))
    (define o (ridge-oracle rc alpha))
    (for ([s (in-list solvers)])
      (test-case (format "~a ridge ~a k ~a" (lcase-name c) s k)
        (check-ridge-fit c #t dt s (ridge-case-fit c dt #:alpha alpha #:solver s) o)))))

(define (en-options r alpha tol max-iter)
  (if (= r 1)
      (hasheq '#:fit lasso-fit '#:alpha alpha '#:tol tol '#:max-iter max-iter)
      (hasheq '#:alpha alpha '#:l1-ratio r '#:tol tol '#:max-iter max-iter)))

(define (en-run-at rc r alpha tol max-iter)
  (match-define (struct* real-case ((case c) [dtype dt])) rc)
  (apply-keywords en-case-fit (en-options r alpha tol max-iter) (list c dt)))

(define (en-oracle rc alpha r)
  (match-define (struct* real-case ([problem p] [outer (cons _ bottom)])) rc)
  (oracle-elastic-net p alpha r #:bottom bottom))

(define (check-en-fit rc o alpha r tol max-iter)
  (match-define (struct* real-case ((case c) [dtype dt] [problem p])) rc)
  (define u (unit-roundoff dt))
  (define run (en-run-at rc r alpha tol max-iter))
  (match-define (struct* en-run ([n-iter k])) run)
  (check-true (< k max-iter) (format "converged: ~a epochs of ~a" k max-iter))
  (check-kkt-laws p
                  run
                  alpha
                  r
                  tol
                  (rows-of c)
                  u
                  #:relative? (not (and (eq? dt 'float32) (< tol default-tol))))
  (when (< tol default-tol)
    (check-distance-laws o run alpha r tol u #t)))

(define (check-real-elastic-net rc)
  (match-define (struct* real-case ((case c) [dtype dt] [problem p])) rc)
  (for ([r (in-list l1-ratios)])
    (define top (exact->inexact (alpha-max p r)))
    (test-case (format "~a l1-ratio ~a above alpha-max" (lcase-name c) r)
      (match-define (struct* en-run ([coef coef]))
        (en-run-at rc r (* top above-alpha-max) default-tol default-max-iter))
      (check-true (andmap zero? coef) (format "coefficients ~a" coef)))
    (for ([k (in-list path-steps)])
      (define alpha (path-alpha top k))
      (define o (en-oracle rc alpha r))
      (for ([tol (list default-tol (tight-tol dt))])
        (test-case (format "~a l1-ratio ~a k ~a tol ~a" (lcase-name c) r k tol)
          (check-en-fit rc
                        o
                        alpha
                        r
                        tol
                        (if (= tol default-tol) default-max-iter tight-max-iter)))))))

(define (check-real-refusal rc)
  (match-define (struct* real-case ((case c) [dtype dt])) rc)
  (define alpha (path-alpha (ridge-lambda-max rc) 0))
  (define o (en-oracle rc alpha 0))
  (test-case (format "~a l1-ratio 0 k 0" (lcase-name c))
    (check-true (< (max-abs (en-solution-coef o)) default-tol))
    (check-cuml-error 'logic
                      #rx"; tol is absolute in the coefficients' units, so scale y up or X down$"
                      (lambda () (en-run-at rc 0 alpha default-tol default-max-iter)))
    (check-en-fit rc o alpha 0 (tight-tol dt) tight-max-iter)))

(define (score-estimators rc)
  (match-define (struct* real-case ([dtype dt] [problem p])) rc)
  (define tol (tight-tol dt))
  (define (path-at r)
    (path-alpha (exact->inexact (alpha-max p r)) 2))
  `((linear-regression ,(linear-regression))
    (ridge ,(ridge #:alpha (ridge-alpha rc 2)))
    (lasso ,(lasso #:alpha (path-at 1) #:tol tol #:max-iter tight-max-iter))
    (elastic-net
     ,(elastic-net #:alpha (path-at 1/2) #:l1-ratio 1/2 #:tol tol #:max-iter tight-max-iter))))

(define (check-real-score rc)
  (match-define (struct* real-case ((case c) [dtype dt])) rc)
  (define X (on-matrix (lcase-X c) dt))
  (define y (on-vector (lcase-y c) dt))
  (define w (on-vector (lcase-w c) dt))
  (for ([entry (in-list (score-estimators rc))])
    (match-define (list name estimator) entry)
    (test-case (format "~a ~a score" (lcase-name c) name)
      (define fitted (fit estimator X y #:sample-weight w))
      (define exact (exact-r2 (lcase-y c) (~> fitted (predict X) device-vector->list) (lcase-w c)))
      (define got (score fitted X y #:sample-weight w))
      (check-true (<= (r2-gap got exact) r2-bound)
                  (format "score ~a, exact ~a" got (exact->inexact exact))))))

(define real-checks
  `((lstsq ,check-real-lstsq "least squares is within the laws' bounds of the exact solve")
    (ridge ,check-real-ridge "ridge is within the laws' bounds of the exact solve")
    (elastic-net ,check-real-elastic-net
                 "lasso and elastic net meet the KKT laws, and the distance laws at a tight tol")
    (score ,check-real-score "each estimator's score is the exact R² of its own predictions")
    (refusal
     ,check-real-refusal
     "l1-ratio 0 at the ridge path's start is refused at the default tol, exact at a tight one")))

(define default-checks '(lstsq ridge elastic-net score))

(define (test-real-set name
                       #:dtypes [dts all-dtypes]
                       #:arms [arms all-arms]
                       #:checks [checks default-checks])
  (for ([check (in-list checks)])
    (match-define (list _ run claim) (assq check real-checks))
    (test-gpu (format "~a ~a ~a: ~a" name dts arms claim)
      (for-each run (real-cases name dts arms)))))

(module+ test
  (require (only-in rackunit check-equal?))
  (define c (lcase "duplicated predictor" '((1 1) (2 2) (4 4)) '(1 2 3) #f))
  (define p (problem-of (lcase-X c) (lcase-y c) (lcase-w c) #t))
  (define-values (inner outer) (eigen-bracket (en-problem-G p)))
  (match-define (solution coef _ kappa)
    (lstsq-solution (real-case c 'float64 p (varying-inner (en-problem-G p) inner) outer)))
  (check-equal? coef '(9/28 9/28))
  (check-equal? kappa +inf.0))
