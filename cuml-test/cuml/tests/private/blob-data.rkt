#lang racket/base

(require (only-in racket/flonum flsingle)
         (only-in racket/math pi))

(provide centred-blobs
         normal-source)

(define (normal-source seed)
  (define g (make-pseudo-random-generator))
  (parameterize ([current-pseudo-random-generator g])
    (random-seed seed))
  (lambda ()
    (parameterize ([current-pseudo-random-generator g])
      (define u (- 1.0 (random)))
      (define v (random))
      (* (sqrt (* -2.0 (log u))) (cos (* 2.0 pi v))))))

(define (centred-blobs centres per-group seed)
  (define next (normal-source seed))
  (values (for*/list ([centre (in-list centres)]
                      [_ (in-range per-group)])
            (for/list ([c (in-list centre)])
              (flsingle (+ c (next)))))
          (for*/list ([k (in-range (length centres))]
                      [_ (in-range per-group)])
            k)))
