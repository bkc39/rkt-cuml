#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters bound-params define-codes define-hyperparameters)
         (only-in cuml/core/private/label-codes prop:label-codes)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in racket/list remove-duplicates)
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
         (only-in "glm.rkt" bounded-count/who intercept-flag/who)
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

(struct logistic-regression-result (coef intercept classes label-type n-iter)
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

(struct label-kind (member? <? =? key unified?))

(define (real-key unify? v)
  (define u
    (if unify?
        (real->double-flonum v)
        v))
  (and (integer? u) (inexact->exact u)))

(define real-labels
  (label-kind real?
              <
              =
              (lambda (unify? v) (and (real? v) (real-key unify? v)))
              (lambda (vs) (ormap inexact? vs))))

(define (named-labels member? <?)
  (label-kind member? <? equal? (lambda (_unify? v) v) (lambda (_vs) #f)))

(define label-kinds (list real-labels (named-labels string? string<?) (named-labels symbol? symbol<?)))

(define (kind-of v)
  (for/first ([kind (in-list label-kinds)]
              #:when ((label-kind-member? kind) v))
    kind))

(define (label-table kind classes)
  (for/hash ([c (in-list classes)]
             [i (in-naturals)])
    (values ((label-kind-key kind) #f c) i)))

(define (host-kind who y)
  (match y
    [(? device-vector?)
     (if (memq (dtype y) label-dtypes)
         #f
         (y-refusal who y))]
    [(cons first _)
     #:when (kind-of first)
     (define kind (kind-of first))
     (for ([v (in-list y)]
           #:unless ((label-kind-member? kind) v))
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
               (flag-value who 'penalty-normalized (ref 'penalty-normalized))
               (class-weight-mode who (ref 'class-weight))))

(define (fit-job who resources X y w keys weights out settings)
  (match-define (qn-settings penalty C ratio flag iterations searches memory tol normalized mode)
    settings)
  (cuml-linear-model-logistic-regression-fit-job who
                                                 resources
                                                 X
                                                 y
                                                 w
                                                 keys
                                                 weights
                                                 out
                                                 penalty
                                                 C
                                                 ratio
                                                 tol
                                                 iterations
                                                 searches
                                                 memory
                                                 flag
                                                 normalized
                                                 mode))

(define (host-labels who kind y)
  (define unify? ((label-kind-unified? kind) y))
  (define keys
    (for/list ([v (in-list y)])
      (or ((label-kind-key kind) unify? v)
          (logic-error "~a: y: expected integral class labels, got ~e" who v))))
  (define sorted (sort (remove-duplicates keys) (label-kind-<? kind)))
  (define table (label-table kind sorted))
  (values (if unify?
              (map exact->inexact sorted)
              sorted)
          (for/list ([key (in-list keys)])
            (hash-ref table key))))

(define (host-weights who kind classes cw)
  (define (key-of c)
    (for/first ([k (in-hash-keys cw)]
                #:when (and ((label-kind-member? kind) k) ((label-kind-=? kind) k c)))
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
  (define (on-device xs)
    (list->device-vector xs #:dtype 'float64 #:resources resources))
  (cond
    [(not (hash? cw)) (values #f #f)]
    [kind
     (values (on-device (for/list ([i (in-range (length classes))])
                          (exact->inexact i)))
             (on-device (host-weights who kind classes cw)))]
    [else
     (values (on-device (for/list ([k (in-hash-keys cw)])
                          (if (real? k)
                              (real->double-flonum k)
                              +nan.0)))
             (on-device (map real->double-flonum (hash-values cw))))]))

(define/who (logistic-regression-fit who
                                     X
                                     y
                                     #:sample-weight [sample-weight #f]
                                     #:hyperparameters logistic-regression-params
                                     #:resources [resources (current-device-resources)])
  (unless (device-matrix? X)
    (logic-error "~a: X: expected a device-matrix, got ~e" who X))
  (define kind (host-kind who y))
  (define settings (qn-settings/who who (bound-params logistic-regression-params)))
  (define w (weight-view who sample-weight))
  (define-values (classes codes)
    (if kind
        (host-labels who kind y)
        (values #f #f)))
  (define y-view
    (if kind
        (list->device-vector codes #:dtype 'int32 #:resources resources)
        y))
  (define-values (keys weights) (weight-vectors kind classes class-weight who resources))
  (define out (deposit-for/who who resources))
  (define n-iter (fit-job who resources X y-view w keys weights out settings))
  (match-define (list coef intercept device-classes) (deposit->list out))
  (logistic-regression-result coef
                              intercept
                              (or classes (device-vector->list device-classes))
                              (if kind
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

(define (applied who result X mode resources)
  (checked-result who result X)
  (match-define (struct* logistic-regression-result
                         ([coef coef] [intercept intercept] [classes classes] [label-type label-type]))
    result)
  (define class-view
    (and (eq? mode 'labels)
         (not (eq? label-type 'list))
         (list->device-vector classes #:dtype label-type #:resources resources)))
  (define out (deposit-for/who who resources))
  (cuml-linear-model-logistic-regression-apply-job who
                                                   resources
                                                   X
                                                   coef
                                                   intercept
                                                   class-view
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
    [(struct* logistic-regression-result ([classes classes] [label-type 'list]))
     (define table (list->vector classes))
     (for/list ([c (in-list (device-vector->list labels))])
       (vector-ref table c))]
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
  (match result
    [(struct* logistic-regression-result ([classes classes] [label-type 'list]))
     (define kind (kind-of (car classes)))
     (define unify? ((label-kind-unified? kind) classes))
     (define table (label-table kind classes))
     (define (code v)
       (hash-ref table ((label-kind-key kind) unify? v) -1))
     (list->device-vector (map code y) #:dtype 'int32 #:resources resources)]
    [(struct* logistic-regression-result ([classes classes] [label-type label-type]))
     (define out (deposit-for/who who resources))
     (cuml-linear-model-label-encode-job
      who
      resources
      y
      (list->device-vector classes #:dtype label-type #:resources resources)
      out)
     (match-define (list codes) (deposit->list out))
     codes]))
