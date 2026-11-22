#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters bound-params define-codes define-hyperparameters)
         (only-in cuml/core/private/label-codes prop:label-codes)
         (only-in cuml/core/private/labels
                  class-view
                  decode-labels
                  device-kind?
                  fit-labels/who
                  label-codes/who
                  label-kind?
                  list-kind?
                  weight-view/who
                  y-kind/who)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in racket/match match match-define struct*)
         (only-in raft current-device-resources device-matrix? device-vector->list shape)
         (only-in raft/unsafe deposit->list)
         (only-in "glm.rkt"
                  bounded-count/who
                  checked-coef/who
                  flag/who
                  intercept-flag/who
                  nonnegative-value/who
                  penalty-code
                  positive-value/who)
         (only-in "native.rkt"
                  cuml-linear-model-logistic-regression-apply-job
                  cuml-linear-model-logistic-regression-fit-job
                  linear-model-label-jobs))

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
  #:guard
  (lambda (coef intercept classes kind n-iter name)
    (if (label-kind? kind)
        (values coef intercept classes kind n-iter)
        (logic-error "~a: label-kind: expected the label kind of a fitted result, got ~e" name kind)))
  #:property prop:label-codes
  (lambda (who result y resources)
    (match-define (struct* logistic-regression-result ([classes classes] [label-kind kind])) result)
    (label-codes/who who linear-model-label-jobs classes kind y resources)))

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

(define-codes class-weight-code [none 0] [balanced 1] [given 2])

(define-codes apply-mode [decision 0] [labels 1] [proba 2] [log-proba 3])

(define (penalty-value who penalty)
  (if (memq penalty '(l1 l2 elasticnet none))
      (penalty-code penalty)
      (logic-error "~a: penalty: expected 'l1, 'l2, 'elasticnet or 'none, got ~e" who penalty)))

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
               (positive-value/who who 'C (ref 'C))
               (ratio-value who (ref 'penalty) (ref 'l1-ratio))
               (intercept-flag/who who (ref 'fit-intercept))
               (bounded-count/who who 'max-iter (ref 'max-iter))
               (bounded-count/who who 'linesearch-max-iter (ref 'linesearch-max-iter))
               (bounded-count/who who 'lbfgs-memory (ref 'lbfgs-memory))
               (nonnegative-value/who who 'tol (ref 'tol))
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

(define/who (logistic-regression-fit who
                                     X
                                     y
                                     #:sample-weight [sample-weight #f]
                                     #:hyperparameters logistic-regression-params
                                     #:resources [resources (current-device-resources)])
  (unless (device-matrix? X)
    (logic-error "~a: X: expected a device-matrix, got ~e" who X))
  (define kind (y-kind/who who y))
  (define settings (qn-settings/who who (bound-params logistic-regression-params)))
  (define w (weight-view/who who sample-weight))
  (match-define (list rows _) (shape X))
  (define-values (y-view classes weights)
    (fit-labels/who who linear-model-label-jobs kind rows y class-weight resources))
  (define out (deposit-for/who who resources))
  (define n-iter (fit-job who resources X y-view w weights out settings))
  (match-define (list coef intercept device-classes) (deposit->list out))
  (logistic-regression-result coef
                              intercept
                              (or classes (device-vector->list device-classes))
                              kind
                              n-iter))

(define (applied who result X mode resources)
  (checked-coef/who who result X logistic-regression-result? logistic-regression-result-coef)
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
     (decode-labels classes labels)]
    [(struct* logistic-regression-result ([label-kind (? device-kind?)])) labels]))

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
