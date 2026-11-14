#lang racket/base

(require (only-in cuml/tests/private/elastic-net-oracle
                  alpha-max
                  en-solution-coef
                  en-solution-intercept
                  oracle-elastic-net
                  problem-of)
         (only-in cuml/tests/private/linear-data
                  lcase-name
                  lcase-w
                  lcase-X
                  lcase-y
                  normal-case
                  oracle-ridge
                  ridge-solution-coef
                  ridge-solution-intercept
                  weights-or-ones)
         (only-in cuml/tests/private/logistic-data
                  blob-case
                  fit-flat
                  lr-case
                  lr-case-name
                  lr-case-w
                  lr-case-X
                  lr-case-y
                  lr-fit-coef
                  relative-gap
                  setting)
         (only-in cuml/tests/private/logistic-newton newton-coef)
         (only-in racket/list append-map first make-list)
         (only-in rackunit check-true test-case)
         (only-in threading ~>)
         (only-in "private/glmnet-data.rkt"
                  glmnet-binomial
                  glmnet-gaussian
                  glmnet-multinomial
                  whole-weighted))

(define gaussian-bound 1e-10)
(define binomial-bound 1e-7)
(define Cs '(0.1 1.0 10.0))

(define gaussian-cases
  (list (normal-case 20 2 0) (whole-weighted (normal-case 20 2 0 #:weighted? #t))))

(define (total-weight c)
  (apply + (weights-or-ones (lcase-w c) (lcase-y c))))

(define (replicated c)
  (define w (weights-or-ones (lr-case-w c) (lr-case-y c)))
  (lr-case (lr-case-name c)
           (append-map make-list w (lr-case-X c))
           (append-map make-list w (lr-case-y c))
           #f))

(define (check-gap name ours exact bound)
  (define gap (relative-gap ours exact))
  (check-true (<= gap bound) (format "~a: gap ~a" name gap)))

(test-case "glmnet-gaussian is the elastic-net optimum at cuML's alpha and l1-ratio"
  (for* ([c (in-list gaussian-cases)]
         [fi (in-list '(#t #f))]
         [ratio (in-list '(1 1/2 1/10))])
    (define p (problem-of (lcase-X c) (lcase-y c) (lcase-w c) fi))
    (define alpha (~> p (alpha-max 1) (* 1/30) exact->inexact))
    (define o (oracle-elastic-net p alpha ratio))
    (check-gap (format "~a l1-ratio ~a intercept ~a" (lcase-name c) ratio fi)
               (fit-flat (glmnet-gaussian c alpha ratio #:fit-intercept fi))
               (append (en-solution-coef o) (list (en-solution-intercept o)))
               gaussian-bound)))

(test-case "glmnet-gaussian at l1-ratio 0 is the ridge optimum at cuML's alpha times W"
  (for* ([c (in-list gaussian-cases)]
         [fi (in-list '(#t #f))])
    (define alpha
      (~> (problem-of (lcase-X c) (lcase-y c) (lcase-w c) fi) (alpha-max 1) (* 1/30) exact->inexact))
    (define o (oracle-ridge (lcase-X c) (lcase-y c) (lcase-w c) (* alpha (total-weight c)) fi))
    (check-gap (format "~a intercept ~a" (lcase-name c) fi)
               (fit-flat (glmnet-gaussian c alpha 0 #:fit-intercept fi))
               (append (ridge-solution-coef o) (list (ridge-solution-intercept o)))
               gaussian-bound)))

(test-case "l2 glmnet-binomial at C is the optimum of the mean log-loss plus 1/(2 C W) |coef|^2"
  (for* ([c (in-list (list (blob-case "binary" 200 4 2 41)
                           (blob-case "binary weighted" 200 4 2 42 #:weighted? #t)))]
         [C (in-list Cs)]
         [fi (in-list '(#t #f))])
    (define rows (replicated c))
    (define lam (/ 1 (* C (length (lr-case-y rows)))))
    (check-gap (format "~a C ~a intercept ~a" (lr-case-name c) C fi)
               (first (lr-fit-coef (glmnet-binomial c (setting 'l2 C #f) #:fit-intercept fi)))
               (newton-coef (lr-case-X rows) (lr-case-y rows) #:lam lam #:fit-intercept fi)
               binomial-bound)))

(test-case "glmnet-multinomial with integer weights is its fit on the replicated rows"
  (define c (blob-case "three blobs weighted" 60 3 3 33 #:weighted? #t))
  (for* ([C (in-list Cs)]
         [penalty (in-list '(l2 l1))])
    (define s (setting penalty C #f))
    (define-values (weighted _) (glmnet-multinomial c s))
    (define-values (rows __) (glmnet-multinomial (replicated c) s))
    (check-gap (format "~a C ~a" penalty C) (fit-flat weighted) (fit-flat rows) 1e-12)))
