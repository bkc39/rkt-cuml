#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters bound-params define-codes define-hyperparameters)
         (only-in cuml/core/private/label-codes prop:label-codes)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in racket/list filter-not remove-duplicates)
         (only-in racket/match match match-define struct*)
         (only-in raft
                  current-device-resources
                  device-matrix?
                  device-vector->list
                  device-vector?
                  dtype
                  list->device-vector
                  shape)
         (only-in raft/unsafe deposit->list)
         (only-in "glm.rkt" bounded-count/who flag/who intercept-flag/who)
         (only-in "native.rkt"
                  cuml-linear-model-label-encode-job
                  cuml-linear-model-logistic-regression-apply-job
                  cuml-linear-model-logistic-regression-fit-job))

(provide (struct-out logistic-regression-result)
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

(struct logistic-regression-result (coef intercept classes label-kind n-iter)
  #:property prop:label-codes
  (lambda (who result y resources) (label-codes who result y resources)))

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

(define-codes class-weight-code [none 0] [balanced 1] [given 2])

(define-codes apply-mode [decision 0] [labels 1] [proba 2] [log-proba 3])

(define label-dtypes '(int32 int64 float32 float64))

(struct label-kind ())
(struct list-kind label-kind (member? <? =? key unified?))
(struct device-kind label-kind (dtype))

(define (key-of kind v #:unify? [unify? #f])
  ((list-kind-key kind) unify? v))

(define (real-key unify? v)
  (define u
    (if unify?
        (real->double-flonum v)
        v))
  (and (integer? u) (inexact->exact u)))

(define real-labels
  (list-kind real?
             <
             =
             (lambda (unify? v) (and (real? v) (real-key unify? v)))
             (lambda (vs) (ormap inexact? vs))))

(define (named-labels member? <?)
  (list-kind member? <? equal? (lambda (_unify? v) v) (lambda (_vs) #f)))

(define label-kinds
  (list real-labels (named-labels string? string<?) (named-labels symbol? symbol<?)))

(define (list-kind-of v)
  (for/first ([kind (in-list label-kinds)]
              #:when ((list-kind-member? kind) v))
    kind))

(define (label-table kind classes)
  (for/hash ([c (in-list classes)]
             [i (in-naturals)])
    (values (key-of kind c) i)))

(define (y-kind who y)
  (match y
    [(? device-vector?)
     (if (memq (dtype y) label-dtypes)
         (device-kind (dtype y))
         (y-refusal who y))]
    [(cons (and first (app list-kind-of (? values kind))) _)
     (for ([v (in-list y)]
           #:unless ((list-kind-member? kind) v))
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
  (define c (and (real? C) (real->double-flonum C)))
  (if (and c (< 0.0 c +inf.0))
      c
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

(define (class-weight-mode who cw)
  (match cw
    [#f (class-weight-code 'none)]
    ['balanced (class-weight-code 'balanced)]
    [(? hash?)
     #:when (for/and ([v (in-hash-values cw)])
              (and (real? v) (positive? v) (< v +inf.0)))
     (class-weight-code 'given)]
    [_
     (logic-error
      "~a: class-weight: expected #f, 'balanced or a hash from labels to positive finite reals, got ~e"
      who
      cw)]))

(define (weight-view who w)
  (if (or (not w) (device-vector? w))
      w
      (logic-error "~a: sample-weight: expected a device-vector or #f, got ~e" who w)))

(struct qn-settings
        (penalty C
                 l1-ratio
                 fit-intercept
                 max-iter
                 linesearch-max-iter
                 lbfgs-memory
                 tol
                 normalized
                 class-weight))

(define (qn-settings/who who params)
  (define (ref key)
    (hash-ref params key))
  (qn-settings (penalty-value who (ref 'penalty))
               (C-value who (ref 'C))
               (ratio-value who (ref 'penalty) (ref 'l1-ratio))
               (intercept-flag/who who (ref 'fit-intercept))
               (bounded-count/who who 'max-iter (ref 'max-iter))
               (bounded-count/who who 'linesearch-max-iter (ref 'linesearch-max-iter))
               (bounded-count/who who 'lbfgs-memory (ref 'lbfgs-memory))
               (tol-value who (ref 'tol))
               (flag/who who 'penalty-normalized (ref 'penalty-normalized))
               (class-weight-mode who (ref 'class-weight))))

(define (fit-job who resources X y w weights out settings)
  (match-define (struct* qn-settings
                         ([penalty penalty] [C C]
                                            [l1-ratio l1-ratio]
                                            [fit-intercept fit-intercept]
                                            [max-iter max-iter]
                                            [linesearch-max-iter linesearch-max-iter]
                                            [lbfgs-memory lbfgs-memory]
                                            [tol tol]
                                            [normalized normalized]
                                            [class-weight class-weight]))
    settings)
  (cuml-linear-model-logistic-regression-fit-job who
                                                 resources
                                                 X
                                                 y
                                                 w
                                                 weights
                                                 out
                                                 penalty
                                                 C
                                                 l1-ratio
                                                 tol
                                                 max-iter
                                                 linesearch-max-iter
                                                 lbfgs-memory
                                                 fit-intercept
                                                 normalized
                                                 class-weight))

(define (host-labels who kind y)
  (define unify? ((list-kind-unified? kind) y))
  (define keys
    (for/list ([v (in-list y)])
      (or (key-of kind v #:unify? unify?)
          (logic-error "~a: y: expected integral class labels, got ~e" who v))))
  (define sorted (sort (remove-duplicates keys) (list-kind-<? kind)))
  (define table (label-table kind sorted))
  (values (if unify?
              (map exact->inexact sorted)
              sorted)
          (map (lambda (key) (hash-ref table key)) keys)))

(define (host-weights who kind classes cw)
  (define (weight-key c)
    (match (for/list ([k (in-hash-keys cw)]
                      #:when (and ((list-kind-member? kind) k) ((list-kind-=? kind) k c)))
             k)
      [(list) #f]
      [(list k) k]
      [(list* a b _) (logic-error "~a: class-weight: keys ~e and ~e both name class ~e" who a b c)]))
  (define missing (filter-not weight-key classes))
  (when (and (pair? missing) (not (= (- (length classes) (length missing)) (hash-count cw))))
    (logic-error "~a: class-weight: classes ~s are not in class-weight" who missing))
  (for/list ([c (in-list classes)])
    (define k (weight-key c))
    (if k
        (real->double-flonum (hash-ref cw k))
        1.0)))

(define (integral-classes y)
  (define integral
    (for/list ([v (in-list (device-vector->list y))]
               #:when (and (rational? v) (integer? v)))
      v))
  (sort (remove-duplicates integral =) <))

(define (class-weight-view who kind classes cw resources)
  (and
   (hash? cw)
   (list->device-vector (host-weights who kind classes cw) #:dtype 'float64 #:resources resources)))

(define (fit-labels who kind y cw resources)
  (match kind
    [(? device-kind?)
     (values y #f (class-weight-view who real-labels (integral-classes y) cw resources))]
    [_
     (define-values (classes codes) (host-labels who kind y))
     (values (list->device-vector codes #:dtype 'int32 #:resources resources)
             classes
             (class-weight-view who kind classes cw resources))]))

(define (class-view kind classes resources)
  (match kind
    [(device-kind dt) (list->device-vector classes #:dtype dt #:resources resources)]
    [_ #f]))

(define/who (logistic-regression-fit who
                                     X
                                     y
                                     #:sample-weight [sample-weight #f]
                                     #:hyperparameters logistic-regression-params
                                     #:resources [resources (current-device-resources)])
  (unless (device-matrix? X)
    (logic-error "~a: X: expected a device-matrix, got ~e" who X))
  (define kind (y-kind who y))
  (define settings (qn-settings/who who (bound-params logistic-regression-params)))
  (define w (weight-view who sample-weight))
  (define-values (y-view classes weights) (fit-labels who kind y class-weight resources))
  (define out (deposit-for/who who resources))
  (define n-iter (fit-job who resources X y-view w weights out settings))
  (match-define (list coef intercept device-classes) (deposit->list out))
  (logistic-regression-result coef
                              intercept
                              (or classes (device-vector->list device-classes))
                              kind
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

(define (applied who result X mode resources)
  (checked-result who result X)
  (match-define (struct* logistic-regression-result
                         ([coef coef] [intercept intercept] [classes classes] [label-kind kind]))
    result)
  (define classes-view (and (eq? mode 'labels) (class-view kind classes resources)))
  (define out (deposit-for/who who resources))
  (cuml-linear-model-logistic-regression-apply-job who
                                                   resources
                                                   X
                                                   coef
                                                   intercept
                                                   classes-view
                                                   (apply-mode mode)
                                                   out)
  (match-define (list v) (deposit->list out))
  v)

(define/who (logistic-regression-decision-function who
                                                   result
                                                   X
                                                   #:resources [resources (current-device-resources)])
  (applied who result X 'decision resources))

(define/who (logistic-regression-predict who
                                         result
                                         X
                                         #:resources [resources (current-device-resources)])
  (define labels (applied who result X 'labels resources))
  (match result
    [(struct* logistic-regression-result ([classes classes] [label-kind (? list-kind?)]))
     (define table (list->vector classes))
     (map (lambda (c) (vector-ref table c)) (device-vector->list labels))]
    [_ labels]))

(define/who (logistic-regression-predict-proba who
                                               result
                                               X
                                               #:resources [resources (current-device-resources)])
  (applied who result X 'proba resources))

(define/who (logistic-regression-predict-log-proba who
                                                   result
                                                   X
                                                   #:resources [resources (current-device-resources)])
  (applied who result X 'log-proba resources))

(define (label-codes who result y resources)
  (match-define (struct* logistic-regression-result ([classes classes] [label-kind kind])) result)
  (match kind
    [(? device-kind?)
     (define out (deposit-for/who who resources))
     (cuml-linear-model-label-encode-job who resources y (class-view kind classes resources) out)
     (match-define (list codes) (deposit->list out))
     codes]
    [_
     (define unify? ((list-kind-unified? kind) classes))
     (define table (label-table kind classes))
     (define (code v)
       (hash-ref table (key-of kind v #:unify? unify?) -1))
     (list->device-vector (map code y) #:dtype 'int32 #:resources resources)]))
