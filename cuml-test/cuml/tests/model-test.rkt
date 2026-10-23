#lang racket/base

(require (only-in cuml/core/private/model
                  deposit->native-model/who
                  live-anchor/who
                  native-model-kind
                  native-model-release!
                  native-model-released?
                  native-model?
                  with-native-models)
         (only-in ffi/unsafe void/reference-sink)
         (only-in racket/async-channel async-channel-put make-async-channel)
         (only-in racket/generator generator yield)
         (only-in racket/match match-define)
         (only-in racket/sandbox call-with-trusted-sandbox-configuration make-evaluator)
         (only-in rackunit check-equal? check-exn check-false check-not-false check-pred check-true)
         (only-in raft
                  current-device-resources
                  device-resources
                  device-vector->list
                  exn:fail:raft?
                  list->device-vector
                  with-device-resources)
         (only-in raft/private/foreign/memory release-failure-count resources-drop-count)
         (only-in raft/tests/private/collect collect-until drain-finalizers!)
         (only-in raft/tests/private/deposits filled)
         (only-in "private/cuml-error.rkt" check-cuml-error raised)
         (only-in "private/model-probe.rkt"
                  await-waiter
                  counts
                  counts-destroyed
                  counts-released
                  probe-counts
                  probe-model
                  probe-release
                  probe-touch
                  probe-touch-job
                  test-model-probe
                  with-gate))

(define MiB (* 1024 1024))
(define gate-seconds 5)

(define (counts- after before)
  (match-define (counts c r d s) after)
  (match-define (counts c0 r0 d0 s0) before)
  (counts (- c c0) (- r r0) (- d d0) (- s s0)))

(define (settled)
  (drain-finalizers!)
  (probe-counts))

(define (since before)
  (counts- (settled) before))

(define (until-destroyed before n)
  (collect-until (lambda () (= (counts-destroyed (probe-counts)) (+ (counts-destroyed before) n)))))

(define released-message "model: expected a live model, got a released one")

(define (check-delta before expected label)
  (check-equal? (since before) expected label))

(define (discard-model)
  (let-values ([(_m _region) (probe-model)])
    (void)))

(test-model-probe
 "a model is released once and destroyed once"
 (define before (settled))
 (define-values (m region) (probe-model))
 (native-model-release! m)
 (native-model-release! m)
 (check-true (native-model-released? m))
 (check-delta before (counts 1 1 0 0) "released twice, the region held")
 (define-values (w w-region) (probe-model))
 (with-native-models ([held w]) (native-model-release! held))
 (check-delta before (counts 2 2 0 0) "released inside with-native-models, then its exit")
 (void/reference-sink m region w w-region)
 (check-true (until-destroyed before 2))
 (check-delta before (counts 2 2 2 0) "released, then dropped")
 (discard-model)
 (check-true (until-destroyed before 3))
 (check-delta before (counts 3 2 3 0) "dropped without a release"))

(test-model-probe
 "a released model is refused, in Racket and in the shim"
 (define-values (m _) (probe-model))
 (define anchor (live-anchor/who 'caller m 'probe-a))
 (native-model-release! m)
 (check-cuml-error 'logic
                   (string-append "caller: " released-message)
                   (lambda () (live-anchor/who 'caller m 'probe-a)))
 (check-cuml-error 'logic
                   (string-append "probe-touch: " released-message)
                   (lambda () (probe-touch m)))
 (check-cuml-error 'logic
                   (string-append "raw-touch: " released-message)
                   (lambda () (probe-touch-job 'raw-touch (current-device-resources) anchor -1))))

(test-model-probe
 "a model of another kind, a non-model and a plain array are refused"
 (define-values (b _) (probe-model #:kind 'probe-b))
 (check-equal? (native-model-kind b) 'probe-b)
 (check-cuml-error 'logic
                   "probe-touch: model: expected a probe-a model, got a probe-b model"
                   (lambda () (probe-touch b)))
 (check-cuml-error
  'logic
  "raw-touch: model: expected a probe-a model, got a probe-b model"
  (lambda ()
    (probe-touch-job 'raw-touch (current-device-resources) (live-anchor/who 'caller b 'probe-b) -1)))
 (check-cuml-error 'logic
                   "probe-touch: model: expected a native model, got 5"
                   (lambda () (probe-touch 5)))
 (check-cuml-error 'logic
                   "raw-touch: model: expected a model anchor, got another array"
                   (lambda ()
                     (probe-touch-job 'raw-touch
                                      (current-device-resources)
                                      (list->device-vector '(1) #:dtype 'int32)
                                      -1)))
 (check-cuml-error
  'logic
  "caller: out: region 0 is not a model anchor"
  (lambda () (deposit->native-model/who 'caller 'probe-a (filled 'float32 3 4) probe-release)))
 (check-cuml-error 'logic
                   "with-native-models: model: expected a native model, got 5"
                   (lambda () (with-native-models ([m 5]) m))))

(test-model-probe "with-native-models checks every binding before it holds any model"
                  (define-values (m region) (probe-model))
                  (check-cuml-error 'logic
                                    "with-native-models: model: expected a native model, got 5"
                                    (lambda () (with-native-models ([a m] [b 5]) 'body)))
                  (check-false (native-model-released? m))
                  (void/reference-sink region))

(define (memory-use)
  (drain-finalizers!)
  (current-memory-use))

(define heap-noise MiB)

(test-model-probe "a model's opaque bytes count toward Racket's memory use until it is released"
                  (define payload (* 64 MiB))
                  (define least (- payload heap-noise))
                  (define before (memory-use))
                  (define-values (m region) (probe-model #:payload-bytes payload))
                  (define held (memory-use))
                  (check-true (>= (- held before) least) (format "grew ~a bytes" (- held before)))
                  (native-model-release! m)
                  (define after (memory-use))
                  (check-true (>= (- held after) least) (format "fell ~a bytes" (- held after)))
                  (void/reference-sink m region))

(define (region-without-its-model payload)
  (define-values (_m region) (probe-model #:payload-bytes payload))
  region)

(test-model-probe
 "a model's opaque bytes count until its last region is collected"
 (define payload (* 64 MiB))
 (define before (settled))
 (define base (memory-use))
 (let ([region (region-without-its-model payload)])
   (define held (- (memory-use) base))
   (check-true (>= held (- payload heap-noise)) (format "the region alone holds ~a bytes" held))
   (check-delta before (counts 1 0 0 0) "the region holds the native model")
   (void/reference-sink region))
 (check-true (until-destroyed before 1))
 (define left (- (memory-use) base))
 (check-true (< left heap-noise) (format "~a bytes left after the region was collected" left))
 (check-delta before (counts 1 0 1 0) "dropping the region destroys the model"))

(define (touched-past-its-resources payload)
  (define drops (resources-drop-count))
  (define-values (m region)
    (with-device-resources ([r (device-resources)])
      (probe-model #:payload-bytes payload #:resources r)))
  (drain-finalizers!)
  (check-equal? (- (resources-drop-count) drops) 1 "the creating resources were released")
  (check-equal? (probe-touch m) payload)
  (native-model-release! m)
  (check-equal? (device-vector->list region) '(0 1 2 3)))

(test-model-probe "a model outlives the Racket resources that created it"
                  (define failures (release-failure-count))
                  (define before (settled))
                  (touched-past-its-resources MiB)
                  (check-true (until-destroyed before 1))
                  (check-delta before (counts 1 1 1 0) "destroyed once, never stale")
                  (check-equal? (- (release-failure-count) failures) 0))

(test-model-probe "published regions outlive a release"
                  (define before (settled))
                  (define-values (m region) (probe-model #:payload-bytes MiB #:region-length 6))
                  (define kept (box region))
                  (native-model-release! m)
                  (check-equal? (device-vector->list (unbox kept)) '(0 1 2 3 4 5))
                  (check-delta before (counts 1 1 0 0) "the region holds the model")
                  (set-box! kept #f)
                  (check-true (until-destroyed before 1))
                  (check-delta before (counts 1 1 1 0) "dropping the region destroys it"))

(define (released-in-order exit-by)
  (define order (make-async-channel))
  (define ((release-logging tag) who resources anchor)
    (async-channel-put order tag)
    (probe-release who resources anchor))
  (define (model tag)
    (define-values (m _) (probe-model #:release (release-logging tag)))
    m)
  (define before (settled))
  (define first (model 'first))
  (define second (model 'second))
  (exit-by first second)
  (check-true (native-model-released? first))
  (check-true (native-model-released? second))
  (check-equal? (counts-released (since before)) 2)
  (list (sync/timeout 0 order) (sync/timeout 0 order) (sync/timeout 0 order)))

(test-model-probe
 "with-native-models releases on every exit, in reverse order"
 (define (normal a b)
   (check-equal? (with-native-models ([x a] [y b]) 'done) 'done))
 (define (raising a b)
   (check-exn #rx"boom" (lambda () (with-native-models ([x a] [y b]) (error 'boom "boom")))))
 (define (escaping a b)
   (check-equal? (let/ec k
                   (with-native-models ([x a] [y b]) (k 'escaped)))
                 'escaped))
 (define (yielding a b)
   (define g
     (generator ()
       (with-native-models ([x a] [y b]) (yield 'inside) 'resumed)))
   (check-equal? (g) 'inside)
   (check-cuml-error 'logic
                     "with-native-models: cannot re-enter its body after its resources were released"
                     g))
 (for ([exit-by (list normal raising escaping yielding)])
   (check-equal? (released-in-order exit-by) '(second first #f) (format "~a" exit-by))))

(test-model-probe
 "a failed release leaves the model live, and a later one succeeds"
 (define before (settled))
 (define-values (m region) (probe-model #:payload-bytes MiB))
 (define released (with-device-resources ([r (device-resources)]) r))
 (define e (raised (lambda () (native-model-release! m #:resources released))))
 (check-pred exn:fail:raft? e)
 (check-equal? (exn-message e)
               "native-model-release!: the device resources on device 0 were released")
 (check-false (native-model-released? m))
 (check-equal? (probe-touch m) MiB)
 (native-model-release! m)
 (check-true (native-model-released? m))
 (check-equal? (counts-released (since before)) 1)
 (void/reference-sink region))

(define syntax-evaluator
  (call-with-trusted-sandbox-configuration
   (lambda () (make-evaluator 'racket/base #:requires '(cuml/core/private/model)))))

(test-model-probe
 "with-native-models refuses a duplicate id and a non-binding"
 (for ([form (list '(with-native-models ([a 1] [a 2]) a)
                   '(with-native-models (5) 1)
                   '(with-native-models ([a]) a)
                   '(with-native-models ([a 1])))])
   (check-exn exn:fail:syntax? (lambda () (syntax-evaluator form)) (format "~s" form)))
 (check-exn #rx"duplicate binding name"
            (lambda () (syntax-evaluator '(with-native-models ([a 1] [a 2]) a)))))

(test-model-probe "a model prints its kind, and whether it is released"
                  (define-values (m _) (probe-model))
                  (check-pred native-model? m)
                  (check-false (native-model? 5))
                  (check-equal? (format "~a" m) "#<native-model probe-a>")
                  (native-model-release! m)
                  (check-equal? (format "~a" m) "#<native-model probe-a released>"))

(test-model-probe "eight threads releasing one model release it once"
                  (define before (settled))
                  (define-values (m region) (probe-model #:payload-bytes MiB))
                  (for-each thread-wait
                            (for/list ([_ (in-range 8)])
                              (thread (lambda () (native-model-release! m)))))
                  (check-true (native-model-released? m))
                  (check-equal? (counts-released (since before)) 1)
                  (void/reference-sink region))

(define raw-refusal (string-append "raw-touch: " released-message))

(define (touch-outcome anchor resources)
  (with-handlers ([exn:fail? (lambda (e) (if (equal? (exn-message e) raw-refusal) 'refused e))])
    (if (= (probe-touch-job 'raw-touch resources anchor -1) MiB) 'touched 'wrong-size)))

(define (touch-racing-release toucher releaser touch-first?)
  (define-values (m region) (probe-model #:payload-bytes MiB))
  (define anchor (live-anchor/who 'caller m 'probe-a))
  (define outcome (make-channel))
  (define (touch)
    (thread (lambda () (channel-put outcome (touch-outcome anchor toucher)))))
  (define (release)
    (thread (lambda () (native-model-release! m #:resources releaser))))
  (define releasing
    (cond
      [touch-first?
       (touch)
       (release)]
      [else
       (begin0 (release)
         (touch))]))
  (define touched (sync/timeout gate-seconds outcome))
  (check-not-false (sync/timeout gate-seconds releasing))
  (check-true (native-model-released? m))
  (void/reference-sink region)
  touched)

(define (tally outcomes kind)
  (length (filter (lambda (o) (eq? o kind)) outcomes)))

(test-model-probe
 "a touch racing a release either finishes or is refused by the shim, and both happen"
 (define toucher (device-resources))
 (define releaser (device-resources))
 (define outcomes
   (for/list ([round (in-range 200)])
     (touch-racing-release toucher releaser (even? round))))
 (define touched (tally outcomes 'touched))
 (define refused (tally outcomes 'refused))
 (printf "race: ~a touched, ~a refused of ~a\n" touched refused (length outcomes))
 (for ([o (in-list outcomes)])
   (check-not-false (memq o '(touched refused)) (format "~e" o)))
 (check-true (positive? touched) "no touch finished before its release")
 (check-true (positive? refused) "no touch was refused by the shim"))

(test-model-probe "a release waits for a gated touch, and finishes once the gate opens"
                  (with-gate (gate open!)
                             (define-values (m region) (probe-model #:payload-bytes MiB))
                             (define touched (make-channel))
                             (thread (lambda () (channel-put touched (probe-touch m #:gate gate))))
                             (check-true (await-waiter gate))
                             (define releaser (thread (lambda () (native-model-release! m))))
                             (check-false (sync/timeout 0.5 releaser)
                                          "the release waits for the touch")
                             (open!)
                             (check-equal? (sync/timeout gate-seconds touched) MiB)
                             (check-not-false (sync/timeout gate-seconds releaser))
                             (check-true (native-model-released? m))
                             (void/reference-sink region)))

(define (killed-while-touching gate)
  (define-values (m region) (probe-model #:payload-bytes MiB))
  (define toucher (thread (lambda () (probe-touch m #:gate gate))))
  (check-true (await-waiter gate))
  (kill-thread toucher)
  (void/reference-sink m region))

(test-model-probe "a killed caller's job holds the model until the job finishes"
                  (define before (settled))
                  (with-gate (gate open!)
                             (killed-while-touching gate)
                             (check-delta before (counts 1 0 0 0) "the gated job holds the model")
                             (open!)
                             (check-true (until-destroyed before 1)))
                  (check-delta before (counts 1 0 1 0) "destroyed once, never stale"))

(define (release-killed-on spawn)
  (define payload (* 64 MiB))
  (define before (settled))
  (with-gate (gate open!)
             (define-values (m region) (probe-model #:payload-bytes payload))
             (define toucher (thread (lambda () (probe-touch m #:gate gate))))
             (check-true (await-waiter gate))
             (define releaser (spawn (lambda () (native-model-release! m))))
             (check-false (sync/timeout 0.3 releaser) "the release waits for the touch")
             (kill-thread releaser)
             (check-false (thread-running? releaser))
             (open!)
             (check-not-false (sync/timeout gate-seconds toucher))
             (define held (memory-use))
             (native-model-release! m)
             (check-true (native-model-released? m))
             (check-equal? (format "~a" m) "#<native-model probe-a released>")
             (check-equal? (counts-released (since before)) 1)
             (define freed (- held (memory-use)))
             (check-true (>= freed (- payload heap-noise)) (format "fell ~a bytes" freed))
             (check-equal? (with-native-models ([x m]) 'ran) 'ran)
             (check-cuml-error 'logic
                               (string-append "probe-touch: " released-message)
                               (lambda () (probe-touch m)))
             (void/reference-sink region)))

(test-model-probe "a release whose thread was killed is finished by the next release"
                  (release-killed-on thread))

(test-model-probe "a release whose thread was suspended by a kill is finished by the next release"
                  (release-killed-on thread/suspend-to-kill))

(test-model-probe
 "a break during a release lets the release finish, then reaches the releasing thread"
 (define payload (* 64 MiB))
 (define before (settled))
 (with-gate (gate open!)
            (define-values (m region) (probe-model #:payload-bytes payload))
            (define toucher (thread (lambda () (probe-touch m #:gate gate))))
            (check-true (await-waiter gate))
            (define outcome (make-channel))
            (define releaser
              (thread (lambda ()
                        (channel-put outcome
                                     (with-handlers ([exn:break? (lambda (_) 'break)])
                                       (native-model-release! m)
                                       'finished)))))
            (check-false (sync/timeout 0.3 releaser) "the release waits for the touch")
            (define held (memory-use))
            (break-thread releaser)
            (open!)
            (check-not-false (sync/timeout gate-seconds toucher))
            (check-equal? (sync/timeout gate-seconds outcome) 'break)
            (check-not-false (sync/timeout gate-seconds releaser))
            (check-true (native-model-released? m))
            (check-equal? (format "~a" m) "#<native-model probe-a released>")
            (check-equal? (counts-released (since before)) 1)
            (define freed (- held (memory-use)))
            (check-true (>= freed (- payload heap-noise)) (format "fell ~a bytes" freed))
            (void/reference-sink region)))
