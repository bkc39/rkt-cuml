#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in racket/list filter-not remove-duplicates)
         (only-in racket/match match match-define)
         (only-in raft device-vector->list device-vector? dtype list->device-vector shape)
         (only-in raft/unsafe deposit->list)
         (only-in "native.rkt"
                  cuml-linear-model-label-classes-job
                  cuml-linear-model-label-encode-job))

(provide class-view
         decode-labels
         device-kind
         device-kind?
         fit-labels/who
         label-codes/who
         label-dtypes
         label-kind?
         list-kind?
         y-kind/who)

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

(define/who (y-kind who y)
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

(define (device-classes who y resources)
  (define out (deposit-for/who who resources))
  (cuml-linear-model-label-classes-job who resources y out)
  (match-define (list classes) (deposit->list out))
  (device-vector->list classes))

(define (y-length y)
  (match y
    [(? list?) (length y)]
    [(app shape (list n)) n]))

(define (class-weighted? who rows y cw)
  (cond
    [(not (hash? cw)) #f]
    [(not (= (y-length y) rows))
     (logic-error "~a: y: expected ~a elements, got ~a" who rows (y-length y))]
    [else #t]))

(define (class-weight-view who kind classes cw resources)
  (list->device-vector (host-weights who kind classes cw) #:dtype 'float64 #:resources resources))

(define/who (fit-labels who kind rows y cw resources)
  (match kind
    [(? device-kind?)
     (values y
             #f
             (and (class-weighted? who rows y cw)
                  (class-weight-view who real-labels (device-classes who y resources) cw resources)))]
    [_
     (define-values (classes codes) (host-labels who kind y))
     (values (list->device-vector codes #:dtype 'int32 #:resources resources)
             classes
             (and (class-weighted? who rows y cw)
                  (class-weight-view who kind classes cw resources)))]))

(define (class-view kind classes resources)
  (match kind
    [(device-kind dt) (list->device-vector classes #:dtype dt #:resources resources)]
    [(? list-kind?) #f]))

(define/who (label-codes who classes kind y resources)
  (match kind
    [(? device-kind?)
     (define out (deposit-for/who who resources))
     (cuml-linear-model-label-encode-job who resources y (class-view kind classes resources) out)
     (match-define (list codes) (deposit->list out))
     codes]
    [(? list-kind?)
     (define unify? ((list-kind-unified? kind) classes))
     (define table (label-table kind classes))
     (define (code v)
       (hash-ref table (key-of kind v #:unify? unify?) -1))
     (list->device-vector (map code y) #:dtype 'int32 #:resources resources)]))

(define (decode-labels classes codes)
  (define table (list->vector classes))
  (map (lambda (c) (vector-ref table c)) (device-vector->list codes)))
