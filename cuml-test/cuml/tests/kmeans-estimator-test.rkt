#lang racket/base

(require (only-in cuml
                  clusterer?
                  estimator-params
                  estimator-set-params
                  estimator?
                  exn:fail:cuml?
                  fit
                  fit-predict
                  fit-transform
                  fitted?
                  kmeans
                  kmeans-model
                  kmeans?
                  predict
                  predictor?
                  score
                  scorer?
                  transform
                  transformer?)
         (only-in cuml/cluster
                  current-batch-samples
                  kmeans-fit
                  kmeans-params
                  kmeans-predict
                  kmeans-result
                  kmeans-result-centroids
                  kmeans-result-inertia
                  kmeans-result-n-iter
                  kmeans-result?
                  kmeans-score
                  kmeans-transform)
         (only-in cuml/cluster/private/kmeans
                  kmeans-fit/who
                  kmeans-predict/who
                  kmeans-score/who
                  kmeans-transform/who)
         (only-in racket/dict in-dict)
         (only-in racket/list make-list)
         (only-in racket/match match match-define)
         (only-in rackunit
                  check-equal?
                  check-false
                  check-not-equal?
                  check-pred
                  check-regexp-match
                  check-true
                  fail
                  test-case)
         (only-in raft device-resources exn:fail:raft-kind exn:fail:raft? shape with-device-resources)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/capabilities.rkt" capabilities)
         (only-in "private/cuml-error.rkt" check-cuml-error raised)
         (only-in "private/kmeans-data.rkt"
                  blank-matrix
                  blob-data
                  cyclic-weights
                  dtypes
                  grid64
                  grid8
                  host
                  host-vector
                  mat
                  scaled-close?
                  start
                  step-weights
                  string-inits
                  tolerance
                  vec))

(define default-params
  (hasheq 'init 'scalable-k-means++
          'n-clusters 8
          'max-iter 300
          'tol 1e-4
          'n-init 'auto
          'random-state 0
          'oversampling-factor 2.0))

(define round-trips
  (list (list '#:init 'init 'random)
        (list '#:n-clusters 'n-clusters 3)
        (list '#:max-iter 'max-iter 10)
        (list '#:tol 'tol 0.5)
        (list '#:n-init 'n-init 2)
        (list '#:random-state 'random-state 7)
        (list '#:oversampling-factor 'oversampling-factor 1.5)))

(define (printed v)
  (list (format "~a" v) (format "~s" v) (format "~v" v)))

(define (outcome r)
  (match-define (kmeans-result centroids labels inertia n-iter) r)
  (list (host centroids) (host-vector labels) inertia n-iter))

(define (grid8-fit [dt 'float64])
  (fit (kmeans #:n-clusters 2 #:random-state 1) (mat grid8 dt)))

(test-case "K1: the default parameters, and every keyword round-trips"
  (check-equal? (estimator-params (kmeans)) default-params)
  (for ([r (in-list round-trips)])
    (match-define (list kw key value) r)
    (check-equal? (hash-ref (estimator-params (keyword-apply kmeans (list kw) (list value) '())) key)
                  value
                  (format "~a" kw))))

(test-case "K1: the estimator's parameters are the table's, without the settings"
  (check-equal? (kmeans-params) default-params)
  (check-equal? (estimator-params (kmeans)) (kmeans-params))
  (check-equal? (estimator-params (kmeans #:max-samples-per-batch 64 #:resources 'r)) (kmeans-params))
  (for ([r (in-list round-trips)])
    (match-define (list kw _ value) r)
    (check-equal? (estimator-params (keyword-apply kmeans (list kw) (list value) '()))
                  (keyword-apply kmeans-params (list kw) (list value) '())
                  (format "~a" kw))))

(test-case "K2: an unknown parameter is refused, and an update is the constructor's"
  (check-cuml-error 'logic
                    "estimator-set-params: kmeans has no parameter bogus"
                    (lambda () (estimator-set-params (kmeans) (hasheq 'bogus 1))))
  (check-cuml-error 'logic
                    "estimator-set-params: kmeans has no parameter \"n-clusters\""
                    (lambda () (estimator-set-params (kmeans) (hash "n-clusters" 3))))
  (check-equal? (estimator-set-params (kmeans) (hasheq 'n-clusters 3)) (kmeans #:n-clusters 3)))

(test-case "K3: an unfitted k-means estimator has every capability and no model"
  (check-equal? (capabilities (kmeans) kmeans?) '(#t #t #t #t #t #t))
  (check-false (kmeans-model (kmeans)))
  (check-false (fitted? (kmeans))))

(test-gpu "K3: a fitted k-means estimator has every capability and a model"
  (define f (grid8-fit))
  (check-equal? (capabilities f kmeans?) '(#t #t #t #t #t #t))
  (check-pred kmeans-result? (kmeans-model f))
  (check-true (fitted? f)))

(test-case "K4: predict, transform and score refuse an unfitted estimator before reading X"
  (for ([(who call) (in-dict (list (cons 'predict (lambda () (predict (kmeans) #f)))
                                   (cons 'transform (lambda () (transform (kmeans) #f)))
                                   (cons 'score (lambda () (score (kmeans) #f)))))])
    (check-cuml-error 'logic (format "~a: kmeans is not fitted" who) call)))

(test-case "K5: an estimator prints the parameters that differ from the defaults"
  (for ([e (in-list (list (kmeans)
                          (kmeans #:random-state 1 #:n-clusters 3)
                          (kmeans #:n-clusters 8)
                          (kmeans #:init 'random)
                          (kmeans #:max-samples-per-batch 64)))]
        [text (in-list '("#<kmeans>" "#<kmeans #:n-clusters 3 #:random-state 1>"
                                     "#<kmeans>"
                                     "#<kmeans #:init 'random>"
                                     "#<kmeans #:max-samples-per-batch 64>"))])
    (check-equal? (printed e) (list text text text))))

(test-gpu "K5: a fitted estimator prints fitted"
  (define text "#<kmeans fitted #:n-clusters 2 #:random-state 1>")
  (check-equal? (printed (grid8-fit)) (list text text text))
  (define bare (fit (kmeans #:n-clusters 2) (mat grid8 'float64)))
  (check-equal? (printed bare) (make-list 3 "#<kmeans fitted #:n-clusters 2>")))

(test-case "K6: equal parameters are equal, with equal hash codes"
  (check-equal? (kmeans #:n-clusters 3) (kmeans #:n-clusters 3))
  (check-equal? (equal-hash-code (kmeans #:n-clusters 3)) (equal-hash-code (kmeans #:n-clusters 3)))
  (check-equal? (equal-secondary-hash-code (kmeans #:n-clusters 3))
                (equal-secondary-hash-code (kmeans #:n-clusters 3)))
  (check-not-equal? (kmeans #:n-clusters 3) (kmeans #:n-clusters 4))
  (check-not-equal? (kmeans) (kmeans #:resources 'other)))

(test-gpu "K6: a fit is equal only to itself"
  (define e (kmeans #:n-clusters 2 #:random-state 1))
  (define f (fit e (mat grid8 'float64)))
  (check-false (fitted? e))
  (check-equal? f f)
  (check-not-equal? e f)
  (check-not-equal? f (fit e (mat grid8 'float64)))
  (check-equal? (estimator-set-params f (hasheq)) e)
  (check-equal? (equal-hash-code (estimator-set-params f (hasheq))) (equal-hash-code e)))

(struct fit-case (name X w kws vals repeatable?))

(define (estimator-of c)
  (keyword-apply kmeans (fit-case-kws c) (fit-case-vals c) '()))

(define (reference-of c)
  (keyword-apply kmeans-fit
                 (fit-case-kws c)
                 (fit-case-vals c)
                 (list (fit-case-X c))
                 #:sample-weight (fit-case-w c)))

(define (weightings dt weights)
  (list #f (vec weights dt)))

(define (grid-cases dt)
  (append (for*/list ([init (in-list string-inits)]
                      [seed (in-range 1 4)]
                      [w (in-list (weightings dt (cyclic-weights 64)))])
            (fit-case (format "grid64 ~a ~a seed ~a~a" dt init seed (if w " weighted" ""))
                      (mat grid64 dt)
                      w
                      '(#:init #:n-clusters #:random-state)
                      (list init 4 seed)
                      #t))
          (for/list ([w (in-list (weightings dt step-weights))])
            (fit-case (format "grid8 ~a matrix init~a" dt (if w " weighted" ""))
                      (mat grid8 dt)
                      w
                      '(#:init)
                      (list (mat start dt))
                      #t))))

(define (blob-cases dt)
  (for*/list ([seed (in-range 1 9)]
              [w (in-list (weightings dt (cyclic-weights 400)))])
    (fit-case (format "blobs ~a seed ~a~a" dt seed (if w " weighted" ""))
              (blob-data 400 3 5 seed dt)
              w
              '(#:n-clusters #:random-state)
              (list 5 seed)
              #f)))

(define (check-same-distances actual expected dt repeatable?)
  (cond
    [repeatable? (check-equal? actual expected)]
    [else
     (define scale (apply max (map (lambda (row) (apply max row)) expected)))
     (check-true (for*/and ([(actual-row expected-row) (in-parallel actual expected)]
                            [(a b) (in-parallel actual-row expected-row)])
                   (scaled-close? a b scale (tolerance dt))))]))

(test-gpu "K7: the estimator's fit, predict, transform and score are leg 1's"
  (for* ([dt (in-list dtypes)]
         [c (in-list (grid-cases dt))])
    (test-case (fit-case-name c)
      (define X (fit-case-X c))
      (define w (fit-case-w c))
      (define f (fit (estimator-of c) X #:sample-weight w))
      (check-equal? (outcome (kmeans-model f)) (outcome (reference-of c)))
      (define C (kmeans-result-centroids (kmeans-model f)))
      (check-equal? (host-vector (predict f X)) (host-vector (kmeans-predict C X)))
      (check-equal? (host (transform f X)) (host (kmeans-transform C X)))
      (check-equal? (score f X) (kmeans-score C X))
      (check-equal? (score f X #:sample-weight w) (kmeans-score C X #:sample-weight w)))))

(test-gpu "K8: fit-predict and fit-transform are predict and transform after fit"
  (for* ([dt (in-list dtypes)]
         [c (in-list (append (grid-cases dt) (blob-cases dt)))])
    (test-case (fit-case-name c)
      (define X (fit-case-X c))
      (define w (fit-case-w c))
      (define e (estimator-of c))
      (define f (fit e X #:sample-weight w))
      (check-equal? (host-vector (fit-predict e X #:sample-weight w)) (host-vector (predict f X)))
      (check-same-distances (host (fit-transform e X #:sample-weight w))
                            (host (transform f X))
                            dt
                            (fit-case-repeatable? c))
      (check-equal? (score f X #:sample-weight w) (- (kmeans-result-inertia (kmeans-model f)))))))

(test-gpu "K9: refitting a fitted estimator fits from its parameters"
  (for* ([dt (in-list dtypes)]
         [init (in-list string-inits)])
    (test-case (format "~a ~a" dt init)
      (define e (kmeans #:n-clusters 4 #:init init #:random-state 1))
      (define B (mat grid64 dt))
      (check-equal? (outcome (kmeans-model (fit (fit e (mat grid8 dt)) B)))
                    (outcome (kmeans-model (fit e B)))))))

(test-gpu "K10: errors name the generic that was called"
  (define X (mat grid8 'float64))
  (define f (grid8-fit))
  (define int32 (blank-matrix 4 2 'int32))
  (for ([message (in-list '("fit: X: 8 samples are fewer than the 9 clusters"
                            "fit: n-clusters: expected at least 1, got 0"
                            "fit: tol: expected greater than 0, got 0"
                            "fit-predict: max-iter: expected at least 0, got -1"
                            "fit: max-samples-per-batch: expected at least 0, got -1"
                            "fit-transform: X: 8 samples are fewer than the 9 clusters"
                            "predict: X: expected float64, got int32"
                            "transform: X: expected float64, got int32"
                            "score: X: expected float64, got int32"))]
        [call (in-list (list (lambda () (fit (kmeans #:n-clusters 9) X))
                             (lambda () (fit (kmeans #:n-clusters 0) X))
                             (lambda () (fit (kmeans #:tol 0) X))
                             (lambda () (fit-predict (kmeans #:max-iter -1) X))
                             (lambda () (fit (kmeans #:max-samples-per-batch -1) X))
                             (lambda () (fit-transform (kmeans #:n-clusters 9) X))
                             (lambda () (predict f int32))
                             (lambda () (transform f int32))
                             (lambda () (score f int32))))])
    (test-case message
      (check-cuml-error 'logic message call)))
  (test-case "an X that is not a device array"
    (match (raised (lambda () (fit (kmeans) '((0 0)))))
      [(? exn:fail:raft? e)
       (check-false (exn:fail:cuml? e))
       (check-regexp-match #rx"^fit:" (exn-message e))]
      [other (fail (format "expected an exn:fail:raft, got ~e" other))])))

(define (check-released message thunk)
  (match (raised thunk)
    [(? exn:fail:raft? e)
     (check-equal? (exn:fail:raft-kind e) 'logic)
     (check-equal? (exn-message e) message)]
    [other (fail (format "expected an exn:fail:raft, got ~e" other))]))

(test-gpu "K11: the resources parameter is forwarded"
  (define X (mat grid8 'float64))
  (define released (with-device-resources ([r (device-resources)]) r))
  (check-released "fit: the device resources on device 0 were released"
                  (lambda () (fit (kmeans #:n-clusters 2 #:resources released) X)))
  (define f
    (with-device-resources ([r (device-resources)])
      (fit (kmeans #:n-clusters 2 #:resources r) X)))
  (for ([(who call) (in-dict (list (cons 'predict (lambda () (predict f X)))
                                   (cons 'transform (lambda () (transform f X)))
                                   (cons 'score (lambda () (score f X)))
                                   (cons 'fit-predict (lambda () (fit-predict f X)))
                                   (cons 'fit-transform (lambda () (fit-transform f X)))))])
    (check-released (format "~a: the device resources on device 0 were released" who) call)))

(define (keywords-of proc)
  (call-with-values (lambda () (procedure-keywords proc)) list))

(define public+who
  (list (cons kmeans-fit kmeans-fit/who)
        (cons kmeans-predict kmeans-predict/who)
        (cons kmeans-transform kmeans-transform/who)
        (cons kmeans-score kmeans-score/who)))

(test-case "K12: leg 1's procedures are their /who forms without the name"
  (check-equal? (keywords-of kmeans-fit)
                '(() (#:init #:max-iter
                             #:max-samples-per-batch
                             #:n-clusters
                             #:n-init
                             #:oversampling-factor
                             #:params
                             #:random-state
                             #:resources
                             #:sample-weight
                             #:tol)))
  (for ([pair (in-list public+who)])
    (match-define (cons public who) pair)
    (define name (object-name public))
    (check-equal? (keywords-of public) (keywords-of who) (format "~a" name))
    (check-equal? (procedure-arity-mask public)
                  (arithmetic-shift (procedure-arity-mask who) -1)
                  (format "~a" name))
    (check-equal? (object-name who) (string->symbol (format "~a/who" name))))
  (check-equal? (map object-name (map car public+who))
                '(kmeans-fit kmeans-predict kmeans-transform kmeans-score)))

(define (exported-names module)
  (define-values (variables syntaxes) (module->exports module))
  (for*/list ([phase+entries (in-list (append variables syntaxes))]
              [entry (in-list (match phase+entries
                                [(cons _phase entries) entries]))])
    (match-define (cons name _origins) entry)
    name))

(define (private-name? name)
  (or (regexp-match? #rx"/(who|proc)$" (symbol->string name))
      (memq name '(init-rows absent fill defaults-for resolve update keys))))

(test-case "K12: cuml/cluster exports no private name, and the table and the batch parameter"
  (define names (exported-names 'cuml/cluster))
  (check-equal? (filter private-name? names) '())
  (check-true (and (memq 'kmeans-params names) (memq 'current-batch-samples names) #t)))

(test-gpu "K13: n-clusters is inferred from a matrix init when the estimator is built"
  (define m (mat start 'float64))
  (define e (kmeans #:init m))
  (check-equal? (hash-ref (estimator-params e) 'n-clusters) (car (shape m)))
  (check-equal? (hash-ref (estimator-params (estimator-set-params e (hasheq 'init 'random)))
                          'n-clusters)
                2))

(define (option-cases rows)
  (for*/list ([k (in-range 1 5)]
              [max-iter (in-range 0 6)]
              [init (in-list (list 'scalable-k-means++ 'k-means++ 'random 'matrix))]
              [n-init (in-range 1 4)]
              [seed (in-range 0 4)]
              #:when (= (modulo (+ k max-iter n-init seed (string-length (format "~a" init))) 7) 0))
    (append (list (cons '#:n-clusters k)
                  (cons '#:max-iter max-iter)
                  (cons '#:n-init n-init)
                  (cons '#:random-state seed))
            (match init
              ['matrix (list (cons '#:init (first-rows-of rows k)))]
              [name (list (cons '#:init name))]))))

(define (first-rows-of rows k)
  (mat (for/list ([row (in-list rows)]
                  [_ (in-range k)])
         row)
       'float64))

(define (kw-call proc options positional)
  (define sorted (sort options keyword<? #:key car))
  (keyword-apply proc (map car sorted) (map cdr sorted) positional))

(define (inferred? options)
  (and (not (assq '#:n-clusters options))
       (match (assq '#:init options)
         [(cons _ (? symbol?)) #f]
         [#f #f]
         [_ #t])))

(define (five-ways X w options)
  (define set (kw-call kmeans-params options '()))
  (define (set-params)
    (fit (kw-call estimator-set-params options (list (kmeans))) X #:sample-weight w))
  (append (list (kw-call kmeans-fit (cons (cons '#:sample-weight w) options) (list X))
                (kmeans-fit X #:params set #:sample-weight w)
                (kmeans-model (fit (kw-call kmeans options '()) X #:sample-weight w))
                (kmeans-model (fit (kmeans set) X #:sample-weight w)))
          (cond
            [(inferred? options)
             (check-cuml-error 'logic "fit: init: expected 8 rows, got 2" set-params)
             '()]
            [else (list (kmeans-model (set-params)))])))

(define (check-five-ways name X w options)
  (match-define (cons reference others) (map outcome (five-ways X w options)))
  (for ([other (in-list others)]
        [way (in-list '(params constructor constructor-set set-params))])
    (check-equal? other reference (format "~a: ~a" name way))))

(test-gpu "K14: five ways of giving the parameters fit the same"
  (for* ([dt (in-list dtypes)]
         [c (in-list (grid-cases dt))])
    (check-five-ways (fit-case-name c)
                     (fit-case-X c)
                     (fit-case-w c)
                     (map cons (fit-case-kws c) (fit-case-vals c))))
  (for ([options (in-list (option-cases grid64))])
    (check-five-ways (format "~a" options) (mat grid64 'float64) #f options)))

(test-gpu "K15: a set's n-clusters stays when only init is given"
  (define X (mat grid8 'float64))
  (define M (mat start 'float64))
  (check-cuml-error 'logic
                    "kmeans-fit: init: expected 8 rows, got 2"
                    (lambda () (kmeans-fit X #:params (kmeans-params) #:init M)))
  (check-equal? (kmeans-result-n-iter (kmeans-fit X #:init M #:max-iter 0)) 0))
