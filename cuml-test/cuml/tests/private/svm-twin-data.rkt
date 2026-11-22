#lang racket/base

(require (for-syntax racket/base)
         (only-in cuml/svm
                  svc-decision-function
                  svc-fit
                  svc-predict
                  svc-result-gamma
                  svc-result-n-iter
                  svc-result-support)
         (only-in cuml/svm/private/native libcuml-version)
         (only-in json json-null)
         (only-in racket/match match-define struct*)
         (only-in racket/runtime-path define-runtime-path)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in "common-data.rkt" worst)
         (only-in "linear-data.rkt" check-law-case law-case on-matrix)
         (only-in "logistic-data.rkt" correct lr-case-X lr-case-y)
         (only-in "python-env.rkt"
                  floats->bits
                  matrix->bits
                  python-sklearn-version
                  test-twin
                  twin-default-params
                  twin-results)
         (only-in "real-logistic-data.rkt" labelled-case labelled-set)
         (only-in "svm-data.rkt" dyadic-blobs svc-result->svm-fit svm-fit))

(provide (struct-out fixture)
         (struct-out trial)
         breast-cancer
         check-cuml-twins
         check-sklearn-dtype
         check-sklearn-twins
         separable-blobs
         sklearn-trials
         svc-twin-params
         trial-fit
         twin-fixtures
         twin-gammas)

(define-runtime-path twin "../python/svc_twin.py")

(struct fixture (name rows labels))

(struct trial (fixture dtype kernel gamma))

(define (blobs n d seed #:sep [sep 3/2] #:flip [flip 1/20])
  (define-values (rows labels) (dyadic-blobs n d seed #:sep sep #:flip flip))
  (fixture (format "blobs ~ax~a sep ~a" n d sep) rows labels))

(define (separable-blobs)
  (blobs 300 4 8 #:sep 8 #:flip 0))

(define (breast-cancer)
  (define set (labelled-set 'breast-cancer "diagnosis" '("id") #:standardise? #t))
  (fixture "breast cancer" (lr-case-X (labelled-case set)) (lr-case-y (labelled-case set))))

(define (twin-fixtures)
  (list (blobs 300 4 5) (blobs 2000 10 6)))

(define twin-gammas '(0.25 auto scale))

(define (gamma->json gamma)
  (if (symbol? gamma)
      (symbol->string gamma)
      (real->double-flonum gamma)))

(define (trial-fit t)
  (match-define (struct* trial
                         ([fixture (struct* fixture ([rows rows] [labels labels]))] [dtype dt]
                                                                                    [kernel kernel]
                                                                                    [gamma gamma]))
    t)
  (svc-fit (on-matrix rows dt) labels #:kernel kernel #:gamma gamma))

(define (twin-request op t gamma)
  (match-define (struct* trial
                         ([fixture (struct* fixture ([rows rows] [labels labels]))] [dtype dt]
                                                                                    [kernel kernel]))
    t)
  (hasheq 'op op
          'dtype (symbol->string dt)
          'X (matrix->bits dt (device-matrix->list* (on-matrix rows dt)))
          'y labels
          'w (json-null)
          'kernel (symbol->string kernel)
          'gamma (gamma->json gamma)
          'degree 3
          'coef0 0.0
          'C 1.0
          'tol 1e-3
          'cache_size 1024.0))

(define (svc-twin-results requests)
  (twin-results twin
                requests
                #:expected-cuml libcuml-version
                #:expected-sklearn python-sklearn-version))

(define (svc-twin-params #:unbound unbound)
  (twin-default-params twin
                       #:unbound unbound
                       #:expected-cuml libcuml-version
                       #:expected-sklearn python-sklearn-version))

(define (requested op trials fits gamma-of)
  (svc-twin-results (for/list ([t (in-list trials)]
                               [r (in-list fits)])
                      (twin-request op t (gamma-of t r)))))

(define (fitted-trials op trials gamma-of)
  (define fits (map trial-fit trials))
  (values fits (requested op trials fits gamma-of)))

(define (svc-result->hash r rows dt)
  (match-define (struct* svm-fit ([sv sv] [dual dual] [b b])) (svc-result->svm-fit r))
  (define X (on-matrix rows dt))
  (hasheq 'support (device-vector->list (svc-result-support r))
          'n_iter (svc-result-n-iter r)
          'predict (svc-predict r X)
          'support_vectors_bits (matrix->bits dt sv)
          'dual_coef_bits (floats->bits dt dual)
          'intercept_bits (floats->bits dt (list b))
          'decision_bits (floats->bits dt (device-vector->list (svc-decision-function r X)))))

(define compared
  '(support n_iter predict support_vectors_bits dual_coef_bits intercept_bits decision_bits))

;; 'scale gamma's distance from Python's in ulps where it is not 0;
;; T1 for those fixtures runs with our gamma passed to Python as a number.
(define scale-ulps (hash '("blobs 300x4 sep 3/2" float64) 1))

(define (scale-ulps-of t)
  (match-define (struct* trial ([fixture (struct* fixture ([name name]))] [dtype dt] [gamma gamma]))
    t)
  (if (eq? gamma 'scale)
      (hash-ref scale-ulps (list name dt) 0)
      0))

(define (ulps gamma theirs)
  (match-define (list ours) (floats->bits 'float64 (list gamma)))
  (match-define (list bits) (hash-ref theirs 'gamma64_bits))
  (abs (- ours bits)))

(define (as-given t _r)
  (trial-gamma t))

(define (check-cuml-twins trials)
  (define-values (fits answers) (fitted-trials "cuml-fit" trials as-given))
  (define pinned
    (if (ormap (lambda (t) (positive? (scale-ulps-of t))) trials)
        (requested "cuml-fit" trials fits resolved)
        answers))
  (for ([t (in-list trials)]
        [r (in-list fits)]
        [theirs (in-list answers)]
        [at-ours (in-list pinned)])
    (match-define (struct* trial
                           ([fixture (struct* fixture ([name name] [rows rows]))] [dtype dt]
                                                                                  [kernel kernel]
                                                                                  [gamma gamma]))
      t)
    (define departs (scale-ulps-of t))
    (test-case (format "~a ~a ~a gamma ~a" name dt kernel gamma)
      (when (eq? gamma 'scale)
        (check-equal? (ulps (svc-result-gamma r) theirs) departs "T2"))
      (define ours (svc-result->hash r rows dt))
      (define reference (if (positive? departs) at-ours theirs))
      (for ([key (in-list compared)])
        (check-equal? (hash-ref ours key) (hash-ref reference key) (format "T1 ~a" key))))))

(define sklearn-kernels '(linear poly rbf))

(define (sklearn-trials fx dt)
  (for/list ([kernel (in-list sklearn-kernels)])
    (trial fx dt kernel 'scale)))

(define sklearn-bounds (hasheq 'linear 0.04 'poly 0.064 'rbf 0.002))

(define sklearn-margin 0.05)

(define (resolved _t r)
  (svc-result-gamma r))

(define (check-sklearn-twins trials #:separable? [separable? #f])
  (define-values (fits answers) (fitted-trials "sklearn-fit" trials resolved))
  (for ([t (in-list trials)]
        [r (in-list fits)]
        [theirs (in-list answers)])
    (match-define (struct* trial
                           ([fixture (struct* fixture ([name name] [rows rows] [labels truth]))]
                            [dtype dt]
                            [kernel kernel]))
      t)
    (test-case (format "~a ~a ~a" name dt kernel)
      (define X (on-matrix rows dt))
      (define ours (device-vector->list (svc-decision-function r X)))
      (define labels (svc-predict r X))
      (define their-decisions (hash-ref theirs 'decision))
      (define their-labels (hash-ref theirs 'predict))
      (cond
        [separable?
         (law-case "S3"
           (check-equal? labels their-labels)
           (check-true (margin-only? (device-vector->list (svc-result-support r))
                                     (hash-ref theirs 'support)
                                     ours
                                     their-decisions)))]
        [else
         (check-law-case "S1" (worst ours their-decisions) (hash-ref sklearn-bounds kernel))
         (law-case "S2"
           (for ([l (in-list labels)]
                 [l-sk (in-list their-labels)]
                 [d (in-list their-decisions)]
                 #:when (> (abs d) sklearn-margin))
             (check-equal? l l-sk)))
         (check-equal? (correct labels truth) (correct their-labels truth) "train accuracy")]))))

(define margin-tol 2e-3)

(define (on-margin? d)
  (<= (abs (- (abs d) 1.0)) margin-tol))

(define (margin-only? ours theirs ds es)
  (for/and ([d (in-list ds)]
            [e (in-list es)]
            [i (in-naturals)]
            #:unless (eq? (and (memv i ours) #t) (and (memv i theirs) #t)))
    (and (on-margin? d) (on-margin? e))))

(define (check-sklearn-dtype dt)
  (test-twin (format
              "standardised breast-cancer ~a decisions are near sklearn's, labels equal off the boundary"
              dt)
    (check-sklearn-twins (sklearn-trials (breast-cancer) dt)))
  (test-twin (format
              "separable ~a blobs give sklearn's labels; the support sets differ only on the margin"
              dt)
    (check-sklearn-twins (sklearn-trials (separable-blobs) dt) #:separable? #t)))
