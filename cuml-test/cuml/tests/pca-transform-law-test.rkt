#lang racket/base

(require (only-in cuml/decomposition pca-fit pca-fit-transform pca-inverse-transform pca-transform)
         (only-in racket/list last make-list take)
         (only-in racket/match match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/pca-data.rkt"
                  column
                  constant-column
                  fitted
                  fitted-mean
                  fitted-variance
                  fitted-of
                  fixture-matrix
                  fixture-name
                  fixture-rows
                  fixtures
                  host-project
                  host-reconstruct
                  integer-fixture
                  norm
                  normal-fixture
                  wide-fixture)
         (only-in "private/program.rkt" run-program)
         (only-in "private/scaler-data.rkt" bits-equal? round-to rows->matrix unit-roundoff))

(define dtypes '(float32 float64))

(define (epsilon dt)
  (* 2 (unit-roundoff dt)))

(define (scales variance dt)
  (for/list ([v (in-list variance)])
    (max (sqrt v) (epsilon dt))))

(define (worst-ratio got want bound)
  (for*/fold ([worst 0.0])
             ([(g-row w-row b-row) (in-parallel got want bound)]
              [(g w b) (in-parallel g-row w-row b-row)])
    (max worst (/ (abs (- g w)) b))))

(define (centred-norms rows mean)
  (for/list ([row (in-list rows)])
    (norm (map - row mean))))

(define (projection-bound rows mean factor columns)
  (for/list ([r (in-list (centred-norms rows mean))])
    (make-list columns (* factor r))))

(define (scaled rows by)
  (for/list ([row (in-list rows)])
    (map * row by)))

(define (fit-cases)
  (for*/list ([dt (in-list dtypes)]
              [f (in-list fixtures)]
              [k (in-list '(#f 3))])
    (list dt f k)))

(test-gpu "L6, L8: transform is (X - mean) C^T and the inverse T C + mean, within their bounds"
  (for ([c (in-list (fit-cases))])
    (match-define (list dt f k) c)
    (test-case (format "~a ~a k=~a" dt (fixture-name f) k)
      (define u (unit-roundoff dt))
      (define rows (fixture-rows f dt))
      (define X (rows->matrix rows dt))
      (define result (pca-fit X #:n-components k))
      (match-define (fitted components _ _ _ mean) (fitted-of result))
      (define d (length mean))
      (define kept (length components))
      (define T (pca-transform result X))
      (define t (device-matrix->list* T))
      (define worst
        (worst-ratio t
                     (host-project rows mean components)
                     (projection-bound rows mean (* 2 d u) kept)))
      (check-true (<= worst 1.0) (format "L6: ~a of the bound" worst))
      (define back (device-matrix->list* (pca-inverse-transform result T)))
      (define inverse-bound
        (for/list ([t-row (in-list t)])
          (for/list ([m (in-list mean)])
            (* 2 kept u (+ (norm t-row) (abs m))))))
      (define worst-inverse (worst-ratio back (host-reconstruct t components mean) inverse-bound))
      (check-true (<= worst-inverse 1.0) (format "L8: ~a of the bound" worst-inverse)))))

(test-gpu "L6, L9: whitening divides each column by max(sqrt variance, epsilon), to unit variance"
  (for ([c (in-list (fit-cases))])
    (match-define (list dt f k) c)
    (test-case (format "~a ~a k=~a" dt (fixture-name f) k)
      (define u (unit-roundoff dt))
      (define rows (fixture-rows f dt))
      (define X (rows->matrix rows dt))
      (define result (pca-fit X #:n-components k #:whiten #t))
      (match-define (fitted components variance _ _ mean) (fitted-of result))
      (define d (length mean))
      (define tw (device-matrix->list* (pca-transform result X)))
      (define worst
        (worst-ratio (scaled tw (scales variance dt))
                     (host-project rows mean components)
                     (projection-bound rows mean (* (+ (* 2 d) 2) u) (length components))))
      (check-true (<= worst 1.0) (format "L6 whitened: ~a of the bound" worst))
      (match-define (list* largest _) variance)
      (for ([v (in-list variance)]
            [j (in-naturals)]
            #:when (>= v (* 1e-3 largest)))
        (define xs (column tw j))
        (define n (length xs))
        (define mu (/ (apply + xs) n))
        (define var (/ (for/sum ([x (in-list xs)]) (* (- x mu) (- x mu))) (sub1 n)))
        (check-true (<= (abs (- var 1.0)) (* 8 d u (/ largest v)))
                    (format "L9: column ~a has variance ~a" j var))))))

(test-gpu "L9: a zero-variance component is divided by epsilon, and the round trip holds"
  (for ([dt (in-list dtypes)])
    (define narrow (round-to dt))
    (define rows
      (for/list ([row (in-list (constant-column 200 4 2 3.0))])
        (map narrow row)))
    (define X (rows->matrix rows dt))
    (define plain (pca-fit X))
    (define whitened (pca-fit X #:whiten #t))
    (match-define (fitted _ variance _ _ _) (fitted-of whitened))
    (check-equal? (last variance) 0.0 (format "~a: the constant column's variance" dt))
    (define t (device-matrix->list* (pca-transform plain X)))
    (define tw (device-matrix->list* (pca-transform whitened X)))
    (define eps (epsilon dt))
    (check-true (bits-equal? (column tw 3)
                             (for/list ([x (in-list (column t 3))])
                               (narrow (/ x eps))))
                (format "~a: the zero-variance column is the projection over epsilon" dt))
    (define back
      (~> (pca-transform whitened X) (pca-inverse-transform whitened _) device-matrix->list*))
    (define u (unit-roundoff dt))
    (match-define (fitted _ _ _ _ mean) (fitted-of whitened))
    (define worst
      (worst-ratio back
                   rows
                   (for/list ([r (in-list (centred-norms rows mean))])
                     (for/list ([m (in-list mean)])
                       (* 4 4 u (+ r (abs m)))))))
    (check-true (<= worst 1.0) (format "~a: round trip ~a of the bound" dt worst))))

(test-gpu "L7: at full rank the round trip returns X, whitened or not"
  (define checked
    (for*/list ([dt (in-list dtypes)]
                [f (in-list fixtures)]
                [whiten (in-list '(#f #t))]
                #:unless (eq? f wide-fixture)
                [rows (in-value (fixture-rows f dt))]
                [result (in-value (pca-fit (rows->matrix rows dt) #:whiten whiten))]
                [lists (in-value (fitted-of result))]
                #:when (>= (last (fitted-variance lists))
                           (* 1e-3 (apply max (fitted-variance lists)))))
      (define u (unit-roundoff dt))
      (define X (rows->matrix rows dt))
      (define mean (fitted-mean lists))
      (define d (length mean))
      (define back
        (~> (pca-transform result X) (pca-inverse-transform result _) device-matrix->list*))
      (define bound
        (for/list ([r (in-list (centred-norms rows mean))])
          (for/list ([m (in-list mean)])
            (* 4 d u (+ r (abs m))))))
      (define worst (worst-ratio back rows bound))
      (check-true (<= worst 1.0)
                  (format "~a ~a ~a: ~a of the bound" dt (fixture-name f) whiten worst))
      (fixture-name f)))
  (check-true (and (member "normal" checked) (member "uniform integers" checked) #t)))

(test-gpu "L10: whitening ignores the row count of the call"
  (for* ([dt (in-list dtypes)]
         [f (in-list (list normal-fixture integer-fixture))])
    (define u (unit-roundoff dt))
    (define rows (fixture-rows f dt))
    (define X (rows->matrix rows dt))
    (define result (pca-fit X #:whiten #t))
    (match-define (fitted components variance _ _ mean) (fitted-of result))
    (define d (length mean))
    (define k (length components))
    (define scale (scales variance dt))
    (define T (pca-transform result X))
    (define tw (device-matrix->list* T))
    (define back (device-matrix->list* (pca-inverse-transform result T)))
    (for ([m (in-list '(1 2 10))])
      (define head (take rows m))
      (define t-head (~> (rows->matrix head dt) (pca-transform result _) device-matrix->list*))
      (define transform-bound
        (for/list ([r (in-list (centred-norms head mean))])
          (for/list ([s (in-list scale)])
            (/ (* 4 d u r) s))))
      (define worst (worst-ratio t-head (take tw m) transform-bound))
      (check-true (<= worst 1.0)
                  (format "~a ~a ~a rows: transform ~a of the bound" dt (fixture-name f) m worst))
      (define back-head
        (~> (take tw m) (rows->matrix dt) (pca-inverse-transform result _) device-matrix->list*))
      (define inverse-bound
        (for/list ([t-row (in-list (take tw m))])
          (for/list ([mu (in-list mean)])
            (* 4 k u (+ (norm (map * t-row scale)) (abs mu))))))
      (define worst-inverse (worst-ratio back-head (take back m) inverse-bound))
      (check-true
       (<= worst-inverse 1.0)
       (format "~a ~a ~a rows: inverse ~a of the bound" dt (fixture-name f) m worst-inverse)))))

(test-gpu "L13a: transform and inverse of one fit repeat bitwise, and fit-transform is the transform of its fit"
  (for* ([dt (in-list dtypes)]
         [whiten (in-list '(#f #t))])
    (define X (fixture-matrix normal-fixture dt))
    (define result (pca-fit X #:whiten whiten #:n-components 5))
    (define T (pca-transform result X))
    (define t (device-matrix->list* T))
    (define back (device-matrix->list* (pca-inverse-transform result T)))
    (for ([_ (in-range 5)])
      (check-true (bits-equal? (device-matrix->list* (pca-transform result X)) t))
      (check-true (bits-equal? (device-matrix->list* (pca-inverse-transform result T)) back)))
    (define-values (again T2) (pca-fit-transform X #:whiten whiten #:n-components 5))
    (check-true (bits-equal? (device-matrix->list* T2)
                             (device-matrix->list* (pca-transform again X))))))

(define fit-program
  `(module fits racket/base
     (require (only-in cuml/decomposition pca-fit)
              (only-in cuml/tests/private/pca-data fitted-of fixture-matrix integer-fixture))
     (write (for*/list ([dt (in-list '(float32 float64))]
                        [solver (in-list '(full jacobi))]
                        [_ (in-range 2)])
              (struct->vector (fitted-of (pca-fit (fixture-matrix integer-fixture dt)
                                                  #:solver solver)))))))

(define (vector->list* v)
  (cond
    [(vector? v) (map vector->list* (vector->list v))]
    [(list? v) (map vector->list* v)]
    [else v]))

(define (pairs xs)
  (let loop ([xs xs])
    (match xs
      [(list* a b more) (cons (list a b) (loop more))]
      [_ '()])))

(test-gpu "L13b: on exact-sum data six fits in three processes are bitwise equal"
  (define outputs
    (for/list ([_ (in-range 3)])
      (match-define (list status out err) (run-program fit-program))
      (check-equal? (list status err) '(0 ""))
      (read (open-input-string out))))
  (match-define (list* first-output _) outputs)
  (for ([o (in-list outputs)])
    (check-true (bits-equal? (vector->list* o) (vector->list* first-output))))
  (for ([pair (in-list (pairs first-output))])
    (match-define (list a b) pair)
    (check-true (equal? a b) "two fits in one process")))
