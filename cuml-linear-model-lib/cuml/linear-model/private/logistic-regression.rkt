#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters define-codes define-hyperparameters)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match match-define)
         (only-in raft
                  current-device-resources
                  device-matrix?
                  device-vector->list
                  device-vector?
                  dtype
                  list->device-vector
                  shape)
         (only-in raft/unsafe deposit->list)
         (only-in "glm.rkt" bounded-count/who intercept-flag/who)
         (only-in "native.rkt"
                  cuml-linear-model-label-encode-job
                  cuml-linear-model-logistic-regression-apply-job
                  cuml-linear-model-logistic-regression-fit-job))

(provide (struct-out logistic-regression-result)
         label-codes/who
         logistic-regression-decision-function
         logistic-regression-decision-function/who
         logistic-regression-fit
         logistic-regression-fit/who
         logistic-regression-params
         logistic-regression-predict
         logistic-regression-predict-log-proba
         logistic-regression-predict-log-proba/who
         logistic-regression-predict-proba
         logistic-regression-predict-proba/who
         logistic-regression-predict/who)

(struct logistic-regression-result (coef intercept classes label-type n-iter))

(define-hyperparameters logistic-regression-params
  [penalty 'l2]
  [C 1.0]
  [l1-ratio #f]
  [fit-intercept #t]
  [max-iter 1000]
  [tol 1e-4]
  [linesearch-max-iter 50]
  [lbfgs-memory 5]
  [penalty-normalized #t]
  [class-weight #f])

(define-codes penalty-code [none 0] [l1 1] [l2 2] [elasticnet 3])

(define label-dtypes '(int32 int64 float32 float64))

(define (label-kind v)
  (cond
    [(real? v) 'real]
    [(string? v) 'string]
    [(symbol? v) 'symbol]
    [else #f]))

(define (label<? kind)
  (match kind
    ['real <]
    ['string string<?]
    ['symbol symbol<?]))

(define (label=? kind)
  (if (eq? kind 'real) = equal?))

(define (label-key kind unify? v)
  (cond
    [(not (eq? kind 'real)) v]
    [(not (real? v)) #f]
    [else
     (define u
       (if unify?
           (real->double-flonum v)
           v))
     (and (integer? u) (inexact->exact u))]))

(define (label-table kind classes)
  (for/hash ([c (in-list classes)]
             [i (in-naturals)])
    (values (label-key kind #f c) i)))

(define (y-kind who y)
  (match y
    [(? device-vector?)
     (if (memq (dtype y) label-dtypes)
         'device
         (y-refusal who y))]
    [(cons first _)
     #:when (label-kind first)
     (define kind (label-kind first))
     (for ([v (in-list y)]
           #:unless (eq? (label-kind v) kind))
       (logic-error "~a: y: expected labels of one kind (reals, strings or symbols), got ~e and ~e"
                    who
                    first
                    v))
     kind]
    [_ (y-refusal who y)]))

(define (y-refusal who y)
  (logic-error
   "~a: y: expected a device-vector of int32, int64, float32 or float64, or a list of reals, strings or symbols, got ~e"
   who
   y))

(define (penalty-value who penalty)
  (if (memq penalty '(l1 l2 elasticnet none))
      (penalty-code penalty)
      (logic-error "~a: penalty: expected 'l1, 'l2, 'elasticnet or 'none, got ~e" who penalty)))

(define (C-value who C)
  (if (and (real? C) (positive? C) (< C +inf.0))
      (real->double-flonum C)
      (logic-error "~a: C: expected a positive finite real, got ~e" who C)))

(define (ratio-value who penalty ratio)
  (cond
    [(eq? penalty 'elasticnet)
     (if (and (real? ratio) (<= 0 ratio 1))
         (real->double-flonum ratio)
         (logic-error "~a: l1-ratio: expected a real from 0 to 1 with 'elasticnet, got ~e"
                      who
                      ratio))]
    [ratio (logic-error "~a: l1-ratio: given with penalty ~e; expected #f" who penalty)]
    [else 0.0]))

(define (tol-value who tol)
  (if (and (real? tol) (>= tol 0) (< tol +inf.0))
      (real->double-flonum tol)
      (logic-error "~a: tol: expected a nonnegative finite real, got ~e" who tol)))

(define (flag-value who key v)
  (match v
    [#t 1]
    [#f 0]
    [_ (logic-error "~a: ~a: expected a boolean, got ~e" who key v)]))

(define (class-weight-mode who cw)
  (match cw
    [#f 0]
    ['balanced 1]
    [(? hash?)
     #:when (for/and ([v (in-hash-values cw)])
              (and (real? v) (positive? v) (< v +inf.0)))
     2]
    [_
     (logic-error
      "~a: class-weight: expected #f, 'balanced or a hash from labels to positive finite reals, got ~e"
      who
      cw)]))

(define (weight-view who w)
  (if (or (not w) (device-vector? w))
      w
      (logic-error "~a: sample-weight: expected a device-vector or #f, got ~e" who w)))

(define (host-labels who kind y)
  (define unify? (and (eq? kind 'real) (ormap inexact? y)))
  (define keys
    (for/list ([v (in-list y)])
      (or (label-key kind unify? v)
          (logic-error "~a: y: expected integral class labels, got ~e" who v))))
  (define sorted (sort (remove-duplicates keys) (label<? kind)))
  (define table (label-table kind sorted))
  (values (if unify?
              (map exact->inexact sorted)
              sorted)
          (for/list ([key (in-list keys)])
            (hash-ref table key))))

(define (host-weights who kind classes cw)
  (define (key-of c)
    (for/first ([k (in-hash-keys cw)]
                #:when (and (eq? (label-kind k) kind) ((label=? kind) k c)))
      k))
  (define missing (filter (lambda (c) (not (key-of c))) classes))
  (when (and (pair? missing) (not (= (- (length classes) (length missing)) (hash-count cw))))
    (logic-error "~a: class-weight: classes ~s are not in class-weight" who missing))
  (for/list ([c (in-list classes)])
    (define k (key-of c))
    (if k
        (real->double-flonum (hash-ref cw k))
        1.0)))

(define (weight-vectors kind classes cw who resources)
  (cond
    [(not (hash? cw)) (values #f #f)]
    [(eq? kind 'device)
     (values (list->device-vector (for/list ([k (in-hash-keys cw)])
                                    (if (real? k)
                                        (real->double-flonum k)
                                        +nan.0))
                                  #:dtype 'float64
                                  #:resources resources)
             (list->device-vector (map real->double-flonum (hash-values cw))
                                  #:dtype 'float64
                                  #:resources resources))]
    [else
     (values (list->device-vector (for/list ([i (in-range (length classes))])
                                    (exact->inexact i))
                                  #:dtype 'float64
                                  #:resources resources)
             (list->device-vector (host-weights who kind classes cw)
                                  #:dtype 'float64
                                  #:resources resources))]))

(define/who (logistic-regression-fit who
                                     X
                                     y
                                     #:sample-weight [sample-weight #f]
                                     #:hyperparameters logistic-regression-params
                                     #:resources [resources (current-device-resources)])
  (unless (device-matrix? X)
    (logic-error "~a: X: expected a device-matrix, got ~e" who X))
  (define kind (y-kind who y))
  (define penalty-number (penalty-value who penalty))
  (define c-value (C-value who C))
  (define ratio (ratio-value who penalty l1-ratio))
  (define flag (intercept-flag/who who fit-intercept))
  (define iterations (bounded-count/who who 'max-iter max-iter))
  (define searches (bounded-count/who who 'linesearch-max-iter linesearch-max-iter))
  (define memory (bounded-count/who who 'lbfgs-memory lbfgs-memory))
  (define tolerance (tol-value who tol))
  (define normalized (flag-value who 'penalty-normalized penalty-normalized))
  (define mode (class-weight-mode who class-weight))
  (define w (weight-view who sample-weight))
  (define-values (host-classes* host-codes)
    (if (eq? kind 'device)
        (values #f #f)
        (host-labels who kind y)))
  (define y-view
    (if host-classes*
        (list->device-vector host-codes #:dtype 'int32 #:resources resources)
        y))
  (define-values (keys values*) (weight-vectors kind host-classes* class-weight who resources))
  (define out (deposit-for/who who resources))
  (define n-iter
    (cuml-linear-model-logistic-regression-fit-job who
                                                   resources
                                                   X
                                                   y-view
                                                   w
                                                   keys
                                                   values*
                                                   out
                                                   penalty-number
                                                   c-value
                                                   ratio
                                                   tolerance
                                                   iterations
                                                   searches
                                                   memory
                                                   flag
                                                   normalized
                                                   mode))
  (match-define (list coef intercept classes) (deposit->list out))
  (logistic-regression-result coef
                              intercept
                              (or host-classes* (device-vector->list classes))
                              (if host-classes*
                                  'list
                                  (dtype y))
                              n-iter))

(define (checked-result who result X)
  (unless (logistic-regression-result? result)
    (raise-argument-error who "logistic-regression-result?" result))
  (unless (device-matrix? X)
    (logic-error "~a: X: expected a device-matrix, got ~e" who X))
  (match-define (list _ p) (shape (logistic-regression-result-coef result)))
  (match-define (list _ d) (shape X))
  (unless (= p d)
    (logic-error "~a: X: expected ~a columns, got ~a" who p d))
  (define fitted (dtype (logistic-regression-result-coef result)))
  (unless (eq? fitted (dtype X))
    (logic-error "~a: X: expected dtype ~a (the fit's), got ~a" who fitted (dtype X))))

(define (applied who result X mode classes resources)
  (checked-result who result X)
  (define out (deposit-for/who who resources))
  (match-define (logistic-regression-result coef intercept _ _ _) result)
  (cuml-linear-model-logistic-regression-apply-job who resources X coef intercept classes mode out)
  (match-define (list v) (deposit->list out))
  v)

(define/who (logistic-regression-decision-function who
                                                   result
                                                   X
                                                   #:resources [resources (current-device-resources)])
  (applied who result X 0 #f resources))

(define/who (logistic-regression-predict who
                                         result
                                         X
                                         #:resources [resources (current-device-resources)])
  (match result
    [(logistic-regression-result _ _ classes 'list _)
     (define codes (applied who result X 1 #f resources))
     (define table (list->vector classes))
     (for/list ([c (in-list (device-vector->list codes))])
       (vector-ref table c))]
    [(logistic-regression-result _ _ classes label-type _)
     (checked-result who result X)
     (applied who
              result
              X
              1
              (list->device-vector classes #:dtype label-type #:resources resources)
              resources)]
    [_ (raise-argument-error who "logistic-regression-result?" result)]))

(define/who (logistic-regression-predict-proba who
                                               result
                                               X
                                               #:resources [resources (current-device-resources)])
  (applied who result X 2 #f resources))

(define/who (logistic-regression-predict-log-proba who
                                                   result
                                                   X
                                                   #:resources [resources (current-device-resources)])
  (applied who result X 3 #f resources))

(define/who (label-codes who result y resources)
  (match result
    [(logistic-regression-result _ _ classes 'list _)
     (define kind (label-kind (car classes)))
     (define unify? (and (eq? kind 'real) (inexact? (car classes))))
     (define table (label-table kind classes))
     (define (code v)
       (hash-ref table (label-key kind unify? v) -1))
     (list->device-vector (map code y) #:dtype 'int32 #:resources resources)]
    [(logistic-regression-result _ _ classes label-type _)
     (define out (deposit-for/who who resources))
     (cuml-linear-model-label-encode-job
      who
      resources
      y
      (list->device-vector classes #:dtype label-type #:resources resources)
      out)
     (match-define (list codes) (deposit->list out))
     codes]))
