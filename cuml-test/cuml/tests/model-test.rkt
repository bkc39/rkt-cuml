#lang racket/base

(require (only-in cuml/core/private/model
                  deposit->native-model
                  live-anchor
                  native-model-kind
                  native-model-release!
                  native-model-released?
                  native-model?
                  with-native-models)
         (only-in ffi/unsafe void/reference-sink)
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

(test-model-probe
 "NM1: a model is released once and destroyed once"
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
 (let-values ([(_m _region) (probe-model)])
   (void))
 (check-true (until-destroyed before 3))
 (check-delta before (counts 3 2 3 0) "dropped without a release"))

(test-model-probe
 "NM2: a released model is refused, in Racket and in the shim"
 (define-values (m _) (probe-model))
 (define anchor (live-anchor 'caller m 'probe-a))
 (native-model-release! m)
 (check-cuml-error 'logic
                   (string-append "caller: " released-message)
                   (lambda () (live-anchor 'caller m 'probe-a)))
 (check-cuml-error 'logic
                   (string-append "probe-touch: " released-message)
                   (lambda () (probe-touch m)))
 (check-cuml-error 'logic
                   (string-append "raw-touch: " released-message)
                   (lambda () (probe-touch-job 'raw-touch (current-device-resources) anchor -1))))

(test-model-probe
 "NM3: a model of another kind, a non-model and a plain array are refused"
 (define-values (b _) (probe-model #:kind 'probe-b))
 (check-equal? (native-model-kind b) 'probe-b)
 (check-cuml-error 'logic
                   "probe-touch: model: expected a probe-a model, got a probe-b model"
                   (lambda () (probe-touch b)))
 (check-cuml-error
  'logic
  "raw-touch: model: expected a probe-a model, got a probe-b model"
  (lambda ()
    (probe-touch-job 'raw-touch (current-device-resources) (live-anchor 'caller b 'probe-b) -1)))
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
 (check-cuml-error 'logic
                   "caller: out: region 0 is not a model anchor"
                   (lambda ()
                     (deposit->native-model 'caller 'probe-a (filled 'float32 3 4) probe-release)))
 (check-cuml-error 'logic
                   "with-native-models: model: expected a native model, got 5"
                   (lambda () (with-native-models ([m 5]) m))))

(define (memory-use)
  (drain-finalizers!)
  (current-memory-use))

(define heap-noise MiB)

(test-model-probe "NM4: a model's opaque bytes count toward Racket's memory use until it is released"
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

(test-model-probe "NM5: a model outlives the Racket resources that created it"
                  (define failures (release-failure-count))
                  (define before (settled))
                  (touched-past-its-resources MiB)
                  (check-true (until-destroyed before 1))
                  (check-delta before (counts 1 1 1 0) "destroyed once, never stale")
                  (check-equal? (- (release-failure-count) failures) 0))

(test-model-probe "NM6: published regions outlive a release"
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
  (define order '())
  (define ((release-logging tag) who resources anchor)
    (set! order (cons tag order))
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
  (reverse order))

(test-model-probe
 "NM7: with-native-models releases on every exit, in reverse order"
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
   (check-equal? (released-in-order exit-by) '(second first) (format "~a" exit-by))))

(test-model-probe
 "NM10: a failed release leaves the model live, and a later one succeeds"
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

(test-model-probe "NM11: with-native-models refuses a duplicate id and a non-binding"
                  (define evaluate
                    (call-with-trusted-sandbox-configuration
                     (lambda () (make-evaluator 'racket/base #:requires '(cuml/core/private/model)))))
                  (for ([form (list '(with-native-models ([a 1] [a 2]) a)
                                    '(with-native-models (5) 1)
                                    '(with-native-models ([a]) a)
                                    '(with-native-models ([a 1])))])
                    (check-exn exn:fail:syntax? (lambda () (evaluate form)) (format "~s" form)))
                  (check-exn #rx"duplicate binding name"
                             (lambda () (evaluate '(with-native-models ([a 1] [a 2]) a)))))

(test-model-probe "printing and predicates"
                  (define-values (m _) (probe-model))
                  (check-pred native-model? m)
                  (check-false (native-model? 5))
                  (check-equal? (format "~a" m) "#<native-model probe-a>")
                  (native-model-release! m)
                  (check-equal? (format "~a" m) "#<native-model probe-a released>"))

(test-model-probe "NM8a: eight threads releasing one model release it once"
                  (define before (settled))
                  (define-values (m region) (probe-model #:payload-bytes MiB))
                  (for-each thread-wait
                            (for/list ([_ (in-range 8)])
                              (thread (lambda () (native-model-release! m)))))
                  (check-true (native-model-released? m))
                  (check-equal? (counts-released (since before)) 1)
                  (void/reference-sink region))

(define (touch-outcome m resources)
  (with-handlers
      ([exn:fail?
        (lambda (e)
          (if (equal? (exn-message e) (string-append "probe-touch: " released-message)) 'refused e))])
    (if (= (probe-touch m #:resources resources) MiB) 'touched 'wrong-size)))

(define (touch-racing-release toucher releaser touch-first?)
  (define-values (m region) (probe-model #:payload-bytes MiB))
  (define touched (box 'not-run))
  (define (touch)
    (thread (lambda () (set-box! touched (touch-outcome m toucher)))))
  (define (release)
    (thread (lambda () (native-model-release! m #:resources releaser))))
  (define threads
    (if touch-first?
        (list (touch) (release))
        (reverse (list (release) (touch)))))
  (for ([t (in-list threads)])
    (check-not-false (sync/timeout gate-seconds t)))
  (check-true (native-model-released? m))
  (void/reference-sink region)
  (unbox touched))

(test-model-probe "NM8b: a touch racing a release either finishes or is refused"
                  (define toucher (device-resources))
                  (define releaser (device-resources))
                  (define outcomes
                    (for/list ([round (in-range 200)])
                      (touch-racing-release toucher releaser (even? round))))
                  (printf "NM8b: ~a touched, ~a refused of ~a\n"
                          (length (filter (lambda (o) (eq? o 'touched)) outcomes))
                          (length (filter (lambda (o) (eq? o 'refused)) outcomes))
                          (length outcomes))
                  (for ([o (in-list outcomes)])
                    (check-true (and (memq o '(touched refused)) #t) (format "~e" o))))

(test-model-probe
 "NM8c: a release waits for a gated touch, and finishes once the gate opens"
 (with-gate (gate open!)
            (define-values (m region) (probe-model #:payload-bytes MiB))
            (define touched (box #f))
            (define toucher (thread (lambda () (set-box! touched (probe-touch m #:gate gate)))))
            (check-true (await-waiter gate))
            (define releaser (thread (lambda () (native-model-release! m))))
            (check-false (sync/timeout 0.5 releaser) "the release waits for the touch")
            (open!)
            (check-not-false (sync/timeout gate-seconds releaser))
            (check-not-false (sync/timeout gate-seconds toucher))
            (check-equal? (unbox touched) MiB)
            (check-true (native-model-released? m))
            (void/reference-sink region)))

(define (killed-while-touching gate)
  (define-values (m region) (probe-model #:payload-bytes MiB))
  (define toucher (thread (lambda () (probe-touch m #:gate gate))))
  (check-true (await-waiter gate))
  (kill-thread toucher)
  (void/reference-sink m region))

(test-model-probe "NM9: a killed caller's job holds the model until the job finishes"
                  (define before (settled))
                  (with-gate (gate open!)
                             (killed-while-touching gate)
                             (check-delta before (counts 1 0 0 0) "the gated job holds the model")
                             (open!)
                             (check-true (until-destroyed before 1)))
                  (check-delta before (counts 1 0 1 0) "destroyed once, never stale"))
