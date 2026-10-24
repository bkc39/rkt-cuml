#lang racket/base

(require (only-in racket/list list-set range shuffle)
         (only-in racket/match match)
         (only-in rackunit check-equal? check-true fail test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/dbscan-data.rkt" pick same-partition? seeded)
         (only-in "private/hdbscan-data.rkt"
                  dataset
                  dataset-names
                  fitted-labels
                  fitted-of
                  fitted-probabilities
                  persistence-close?
                  same-fit?)
         (only-in "private/program.rkt" run-program))

(define (fitted->list f)
  (cdr (vector->list (struct->vector f))))

(define (without-persistence fields)
  (list-set fields 2 '()))

(define (close? a b tolerance)
  (and (= (length a) (length b))
       (for/and ([x (in-list a)]
                 [y (in-list b)])
         (<= (abs (- x y)) (* tolerance (max (abs x) (abs y) 1e-30))))))

(define determinism-program
  `(module determinism racket/base
     (require (only-in cuml/tests/private/hdbscan-data dataset dataset-names fitted-of))
     (write (for/list ([name (in-list dataset-names)])
              (cdr (vector->list (struct->vector (fitted-of (dataset name)))))))))

(test-gpu "HD4: five fits are bit-identical, persistence within 1e-5"
  (for ([name (in-list dataset-names)])
    (test-case (symbol->string name)
      (define rows (dataset name))
      (define first-fit (fitted-of rows))
      (for ([_ (in-range 4)])
        (check-true (same-fit? (fitted-of rows) first-fit))))))

(test-gpu "HD4: another process fits the same trees"
  (match (run-program determinism-program)
    [(list 0 output _)
     (for ([name (in-list dataset-names)]
           [theirs (in-list (read (open-input-string output)))])
       (test-case (symbol->string name)
         (define ours (fitted->list (fitted-of (dataset name))))
         (check-equal? (without-persistence theirs) (without-persistence ours))
         (check-true (persistence-close? (list-ref theirs 2) (list-ref ours 2)))))]
    [(list status _ errors) (fail (format "the fit's process exited with ~a:\n~a" status errors))]))

(define (noise-set labels)
  (for/list ([l (in-list labels)]
             [i (in-naturals)]
             #:when (= l -1))
    i))

(define (permuted-fits name seed)
  (define rows (dataset name))
  (define perm (seeded seed (lambda () (shuffle (range (length rows))))))
  (values perm (fitted-of rows) (fitted-of (pick rows perm))))

(define order-dependent '(grid30 one-blob))

(test-gpu "HD5: permuting rows permutes the partition and the noise set, but on tied grid30 and one-blob"
  (for* ([name (in-list (remq* order-dependent dataset-names))]
         [seed (in-range 2)])
    (test-case (format "~a, seed ~a" name seed)
      (define-values (perm base permuted) (permuted-fits name seed))
      (define labels (fitted-labels permuted))
      (check-true (same-partition? (fitted-labels base) perm labels (range (length perm))))
      (check-equal? (noise-set labels) (noise-set (pick (fitted-labels base) perm))))))

(test-gpu "HD5: probabilities permute within 1e-6"
  (for ([name (in-list '(b50 b10 moons circles))])
    (test-case (symbol->string name)
      (define-values (perm base permuted) (permuted-fits name 0))
      (check-true
       (close? (fitted-probabilities permuted) (pick (fitted-probabilities base) perm) 1e-6)))))

(test-gpu "HD6: X*4+3 gives the same labels"
  (for ([name (in-list '(b50 b10 moons))])
    (test-case (symbol->string name)
      (define rows (dataset name))
      (check-equal? (fitted-labels (fitted-of (for/list ([row (in-list rows)])
                                                (map (lambda (x) (+ (* 4.0 x) 3.0)) row))))
                    (fitted-labels (fitted-of rows))))))
