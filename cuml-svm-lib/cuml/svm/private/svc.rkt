#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters bound-params define-codes define-hyperparameters)
         (only-in cuml/core/private/label-codes prop:label-codes)
         (only-in cuml/core/private/labels
                  class-codes/who
                  class-view
                  decode-labels
                  label-codes/who
                  label-kind?
                  list-kind?
                  y-kind/who)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in racket/match match match-define struct*)
         (only-in raft
                  current-device-resources
                  device-matrix?
                  device-vector?
                  dtype
                  shape)
         (only-in raft/unsafe deposit->list)
         (only-in "kernel.rkt"
                  kernel-code
                  svm-kernel
                  svm-settings/who
                  svm-solver)
         (only-in "native.rkt" cuml-svm-svc-apply-job cuml-svm-svc-fit-job svm-label-jobs))

(provide (struct-out svc-result)
         svc-decision-function
         svc-decision-function/who
         svc-fit
         svc-fit/who
         svc-params
         svc-predict
         svc-predict/who)

(struct svc-result
        (support-vectors support
                         dual-coef
                         intercept
                         n-support
                         classes
                         label-kind
                         n-iter
                         kernel
                         degree
                         gamma
                         coef0
                         cache-size)
  #:guard
  (lambda (sv s dual b n-support classes kind n-iter kernel degree gamma coef0 cache-size name)
    (if (label-kind? kind)
        (values sv s dual b n-support classes kind n-iter kernel degree gamma coef0 cache-size)
        (logic-error "~a: label-kind: expected the label kind of a fitted result, got ~e" name kind)))
  #:property prop:label-codes
  (lambda (who result y resources)
    (match-define (struct* svc-result ([classes classes] [label-kind kind])) result)
    (label-codes/who who svm-label-jobs classes kind y resources)))

(define-hyperparameters svc-params
  [kernel 'rbf]
  [degree 3]
  [gamma 'scale]
  [coef0 0.0]
  [C 1.0]
  [tol 1e-3]
  [cache-size 1024.0]
  [max-iter #f]
  [nochange-steps 1000])

(define float-dtypes '(float32 float64))

(define (float-matrix who X)
  (if (and (device-matrix? X) (memq (dtype X) float-dtypes))
      X
      (logic-error "~a: X: expected a device-matrix of float32 or float64, got ~e" who X)))

(define (weight-view who w)
  (if (or (not w) (device-vector? w))
      w
      (logic-error "~a: sample-weight: expected a device-vector or #f, got ~e" who w)))

(define (y-length y)
  (match y
    [(? list?) (length y)]
    [(app shape (list n)) n]))

(define ((two-classes who) classes)
  (match classes
    [(list _ _) (void)]
    [(list only) (logic-error "~a: y: expected two classes, got one (~e)" who only)]
    [(list* a b _)
     (logic-error "~a: y: expected two classes, got ~a (~e ~e ...); multiclass svc is not supported yet"
                  who
                  (length classes)
                  a
                  b)]))

(define/who (svc-fit who
                     X
                     y
                     #:sample-weight [sample-weight #f]
                     #:hyperparameters svc-params
                     #:resources [resources (current-device-resources)])
  (float-matrix who X)
  (define kind (y-kind/who who y))
  (define params (bound-params svc-params))
  (define-values (kernel-settings solver-settings) (svm-settings/who who params))
  (match-define (svm-kernel code degree* mode gamma* coef0* cache) kernel-settings)
  (match-define (svm-solver C* tol* max-iter* nochange) solver-settings)
  (define w (weight-view who sample-weight))
  (match-define (list rows _) (shape X))
  (unless (= (y-length y) rows)
    (logic-error "~a: y: expected ~a elements, got ~a" who rows (y-length y)))
  (define-values (classes codes) (class-codes/who who svm-label-jobs kind y (two-classes who) resources))
  (define out (deposit-for/who who resources))
  (define-values (resolved n-iter negatives positives empty)
    (cuml-svm-svc-fit-job who
                          resources
                          X
                          codes
                          w
                          out
                          code
                          degree*
                          mode
                          gamma*
                          coef0*
                          C*
                          tol*
                          cache
                          max-iter*
                          nochange))
  (match-define (list negative positive) classes)
  (unless (= empty -1)
    (logic-error "~a: sample-weight: class ~e has no row of positive weight"
                 who
                 (if (zero? empty) negative positive)))
  (match-define (list support-vectors support dual intercept) (deposit->list out))
  (svc-result support-vectors
              support
              dual
              intercept
              (list negatives positives)
              classes
              kind
              n-iter
              kernel
              degree*
              resolved
              coef0*
              cache))

(define (checked-result who result X)
  (unless (svc-result? result)
    (logic-error "~a: result: expected an svc-result, got ~e" who result))
  (match-define (struct* svc-result ([support-vectors sv])) result)
  (match-define (list _ columns) (shape sv))
  (unless (and (device-matrix? X) (eq? (dtype X) (dtype sv)))
    (logic-error "~a: X: expected a device-matrix of ~a, got ~e" who (dtype sv) X))
  (match-define (list _ given) (shape X))
  (unless (= given columns)
    (logic-error "~a: X: expected ~a columns, got ~a" who columns given)))

(define-codes apply-mode [decision 0] [labels 1])

(define (applied who result X mode resources)
  (checked-result who result X)
  (match-define (struct* svc-result ([support-vectors sv] [dual-coef dual]
                                                          [intercept intercept]
                                                          [classes classes]
                                                          [label-kind kind]
                                                          [kernel kernel]
                                                          [degree degree]
                                                          [gamma gamma]
                                                          [coef0 coef0]
                                                          [cache-size cache]))
    result)
  (define classes-view (and (eq? mode 'labels) (class-view kind classes resources)))
  (define out (deposit-for/who who resources))
  (cuml-svm-svc-apply-job who
                          resources
                          X
                          sv
                          dual
                          intercept
                          classes-view
                          out
                          (kernel-code kernel)
                          degree
                          gamma
                          coef0
                          cache
                          (apply-mode mode))
  (match-define (list v) (deposit->list out))
  v)

(define/who (svc-decision-function who result X #:resources [resources (current-device-resources)])
  (applied who result X 'decision resources))

(define/who (svc-predict who result X #:resources [resources (current-device-resources)])
  (define labels (applied who result X 'labels resources))
  (match result
    [(struct* svc-result ([classes classes] [label-kind (? list-kind?)])) (decode-labels classes labels)]
    [_ labels]))
