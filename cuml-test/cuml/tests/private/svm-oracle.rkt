#lang racket/base

(require (only-in cuml/svm svc-decision-function)
         (only-in math/bigfloat bf bf-precision bfexp bftanh bigfloat->rational)
         (only-in racket/flonum flsingle)
         (only-in racket/list count first)
         (only-in racket/match match match-define struct*)
         (only-in rackunit check-true test-case)
         (only-in raft device-vector->list)
         (only-in threading ~>)
         (only-in "common-data.rkt" gamma unit-roundoff)
         (only-in "exact-algebra.rkt" dot)
         (only-in "linear-data.rkt" check-law-case law-case on-matrix)
         (only-in "svm-data.rkt"
                  case-fit
                  fresh-rows
                  kkt-cases
                  svc-result->svm-fit
                  svm-case
                  svm-case-dtype
                  svm-case-kernel
                  svm-case-name
                  svm-fit
                  weighted-cases))

(provide (struct-out kkt)
         check-kkt
         check-referee
         check-weighted-kkt
         decision-bound
         decision-ratio
         exact-decision
         exact-kernel
         kkt-bounds
         kkt-of
         penalties
         signed-labels)

(struct kkt
        (signs-agree box-excess
                     balance
                     gap
                     violation
                     zero-weight-alpha
                     support-exact
                     rows-exact
                     counts)
  #:transparent)

(define (exact v)
  (inexact->exact v))

(define (abs-dot a b)
  (for/sum ([x (in-list a)] [y (in-list b)]) (abs (* x y))))

(define (transcendental f q)
  (parameterize ([bf-precision 192])
    (bigfloat->rational (f (bf q)))))

(define (exact-kernel kernel g degree coef0)
  (define γ (exact g))
  (define c₀ (exact coef0))
  (match kernel
    ['linear dot]
    ['poly (lambda (a b) (expt (+ (* γ (dot a b)) c₀) degree))]
    ['rbf
     (lambda (a b)
       (transcendental bfexp (- (* γ (for/sum ([x (in-list a)] [y (in-list b)]) (expt (- x y) 2))))))]
    ['sigmoid (lambda (a b) (transcendental bftanh (+ (* γ (dot a b)) c₀)))]))

(define (fit-kernel fit)
  (match-define (struct* svm-fit ([kernel kernel] [gamma g] [degree degree] [coef0 coef0])) fit)
  (exact-kernel kernel g degree coef0))

(define (exact-rows rows)
  (for/list ([row (in-list rows)])
    (map exact row)))

(define (signed-labels classes y)
  (match-define (list _ positive) classes)
  (for/list ([v (in-list y)])
    (if (equal? v positive) 1 -1)))

(define (rounded dt v)
  (define narrow (if (eq? dt 'float32) flsingle values))
  (~> v exact->inexact narrow exact))

(define (penalties c)
  (match-define (struct* svm-case ([C C] [w w] [y y] [dtype dt])) c)
  (define CT (rounded dt C))
  (for/list ([wi (in-list (or w (map (lambda (_) 1) y)))])
    (rounded dt (* CT wi))))

(define (kkt-bounds dt tol)
  (if (eq? dt 'float32)
      (values (+ 1 (expt 2 -23)) (* 2 (exact tol)))
      (values 1 (* (exact tol) (+ 1 (expt 2 -20))))))

(define (spread-duals n support dual)
  (define full (make-vector n 0))
  (for ([i (in-list support)]
        [v (in-list dual)])
    (vector-set! full i (exact v)))
  full)

(define (balance-bound dt n-iter Cs dual total)
  (define u (exact (unit-roundoff dt)))
  (+ (* 2 u n-iter (apply max Cs)) (* (gamma (length dual) u) total)))

(define (kkt-of c fit)
  (match-define (struct* svm-case ([X X₀] [y y₀] [w w] [dtype dt])) c)
  (match-define (struct* svm-fit
                         ([sv sv₀] [support support]
                                   [dual dual₀]
                                   [b b₀]
                                   [classes classes]
                                   [n-support n-support]
                                   [n-iter n-iter]))
    fit)
  (define X (exact-rows X₀))
  (define sv (exact-rows sv₀))
  (define dual (map exact dual₀))
  (define b (exact b₀))
  (define ys (signed-labels classes y₀))
  (define Cs (penalties c))
  (define n (length X))
  (define k (fit-kernel fit))
  (define duals (spread-duals n support dual))
  (define alphas
    (for/list ([y (in-list ys)]
               [i (in-naturals)])
      (* y (vector-ref duals i))))
  (define fs
    (for/list ([x (in-list X)]
               [y (in-list ys)])
      (- (for/sum ([s (in-list sv)] [d (in-list dual)]) (* d (k s x))) y)))
  (define (in-up? y a C)
    (or (and (> y 0) (< a C)) (and (< y 0) (> a 0))))
  (define (in-low? y a C)
    (or (and (< y 0) (< a C)) (and (> y 0) (> a 0))))
  (define (over pred)
    (for/list ([f (in-list fs)]
               [y (in-list ys)]
               [a (in-list alphas)]
               [C (in-list Cs)]
               #:when (pred y a C))
      f))
  (define highest-low (apply max (over in-low?)))
  (define lowest-up (apply min (over in-up?)))
  (define total (for/sum ([d (in-list dual)]) (abs d)))
  (define signs (list->vector ys))
  (define rows (list->vector X₀))
  (define signs-agree
    (for/and ([i (in-list support)]
              [d (in-list dual)])
      (= (if (positive? d) 1 -1) (vector-ref signs i))))
  (define box-excess
    (for/fold ([worst 0])
              ([a (in-list alphas)]
               [C (in-list Cs)]
               #:when (positive? C))
      (max worst (/ a C))))
  (define zero-weight-alpha
    (if w
        (for/sum ([a (in-list alphas)] [wi (in-list w)] #:when (zero? wi)) (abs a))
        0))
  (define lengths-agree (= (length sv) (length support) (length dual) (apply + n-support)))
  (define support-exact
    (and lengths-agree (andmap (lambda (d) (not (zero? d))) dual) (apply < support)))
  (define rows-exact
    (and lengths-agree
         (for/and ([s (in-list sv)]
                   [i (in-list support)])
           (equal? s (map (lambda (v) (rounded dt v)) (vector-ref rows i))))))
  (define counts (equal? n-support (list (count negative? dual) (count positive? dual))))
  (define balance (/ (abs (apply + dual)) (balance-bound dt n-iter Cs dual total)))
  (define gap (- highest-low lowest-up))
  (define violation (max (+ highest-low b) (- (- b) lowest-up)))
  (kkt signs-agree
       box-excess
       balance
       gap
       violation
       zero-weight-alpha
       support-exact
       rows-exact
       counts))

(define (exact-decision fit x)
  (match-define (struct* svm-fit ([sv sv] [dual dual] [b b])) fit)
  (define k (fit-kernel fit))
  (define ex (map exact x))
  (+ (for/sum ([s (in-list sv)] [d (in-list dual)]) (* (exact d) (k (map exact s) ex))) (exact b)))

(define (kernel-error fit u s x)
  (match-define (struct* svm-fit ([kernel kernel] [gamma g] [degree degree] [coef0 coef0])) fit)
  (define γ (exact g))
  (define c₀ (abs (exact coef0)))
  (define d (length x))
  (define sx (abs-dot s x))
  (define eₐ (+ (* γ (gamma (+ d 1) u) sx) (* u c₀)))
  (match kernel
    ['linear (* (gamma d u) sx)]
    ['poly
     (define a (abs (+ (* γ (dot s x)) (exact coef0))))
     (+ (* degree (expt (+ a eₐ) (max 0 (- degree 1))) eₐ) (* 4 u (expt a degree)))]
    ['rbf
     (define eᵣ (* γ (gamma (+ d 3) u) (+ (dot x x) (dot s s) (* 2 sx))))
     (define k ((fit-kernel fit) s x))
     (* k (+ eᵣ (* eᵣ eᵣ) (* 4 u)))]
    ['sigmoid (+ eₐ (* 4 u))]))

(define (decision-bound fit dt x)
  (match-define (struct* svm-fit ([sv sv] [dual dual] [b b])) fit)
  (define u (exact (unit-roundoff dt)))
  (define k (fit-kernel fit))
  (define ex (map exact x))
  (define rows (map (lambda (s) (map exact s)) sv))
  (+ (for/sum ([s (in-list rows)] [d (in-list dual)]) (* (abs (exact d)) (kernel-error fit u s ex)))
     (* (gamma (+ (length sv) 1) u)
        (+ (for/sum ([s (in-list rows)] [d (in-list dual)]) (abs (* (exact d) (k s ex))))
           (abs (exact b))))))

(define (decision-ratio fit dt rows decisions)
  (for/fold ([worst 0])
            ([x (in-list rows)]
             [d (in-list decisions)])
    (define gap (abs (- (exact d) (exact-decision fit x))))
    (define bound (decision-bound fit dt x))
    (max worst
         (cond
           [(positive? bound) (/ gap bound)]
           [(zero? gap) 0]
           [else +inf.0]))))

(define fresh 200)

(define (check-referee kernels #:dtypes [dtypes '(float32 float64)] #:half [half #f])
  (define chosen
    (for/list ([c (in-list (kkt-cases))]
               #:when (memq (svm-case-kernel c) kernels)
               #:when (memq (svm-case-dtype c) dtypes))
      c))
  (for ([c (in-list chosen)]
        [i (in-naturals)]
        #:when (or (not half) (= (modulo i 2) half)))
    (match-define (struct* svm-case ([X X] [dtype dt])) c)
    (test-case (svm-case-name c)
      (define r (case-fit c))
      (define rows (fresh-rows fresh (length (first X)) (length X)))
      (define decisions (device-vector->list (svc-decision-function r (on-matrix rows dt))))
      (check-law-case "R1" (decision-ratio (svc-result->svm-fit r) dt rows decisions) 1))))

(define kkt-tol 1e-3)

(define (check-kkt c [fit (svc-result->svm-fit (case-fit c))])
  (match-define (struct* svm-case ([dtype dt])) c)
  (define-values (box gap-bound) (kkt-bounds dt kkt-tol))
  (match-define (struct* kkt
                         ([signs-agree signs] [box-excess excess]
                                              [balance ratio]
                                              [gap gap]
                                              [violation violation]
                                              [zero-weight-alpha zero-alpha]
                                              [support-exact support]
                                              [rows-exact rows]
                                              [counts counts]))
    (kkt-of c fit))
  (law-case "K1 signs"
    (check-true signs))
  (check-law-case "K1 box" excess box)
  (check-law-case "K2" ratio 1)
  (check-law-case "K3" gap gap-bound)
  (check-law-case "K4" violation gap-bound)
  (law-case "K5"
    (check-true (zero? zero-alpha)))
  (law-case "K6"
    (check-true support)
    (check-true counts))
  (law-case "K7"
    (check-true rows)))

(define (check-weighted-kkt dt)
  (for ([c (in-list (weighted-cases))]
        #:when (eq? (svm-case-dtype c) dt))
    (test-case (svm-case-name c)
      (check-kkt c))))
