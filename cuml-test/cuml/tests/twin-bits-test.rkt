#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in "private/python-env.rkt" bits->floats floats->bits matrix->bits))

(define rows
  `(("floats->bits, exact float32" ,floats->bits
                                   float32
                                   (1 1/2 -2)
                                   (#x3f800000 #x3f000000 #xc0000000))
    ("floats->bits, a float32 rounding" ,floats->bits float32 (1/3) (#x3eaaaaab))
    ("floats->bits, float64" ,floats->bits
                             float64
                             (1/3 1.5 -0.0)
                             (#x3fd5555555555555 #x3ff8000000000000 #x8000000000000000))
    ("matrix->bits, row-major" ,matrix->bits
                               float32
                               ((1 2) (3 4))
                               (#x3f800000 #x40000000 #x40400000 #x40800000))
    ("matrix->bits, float64" ,matrix->bits
                             float64
                             ((1/2) (-2))
                             (#x3fe0000000000000 #xc000000000000000))
    ("floats->bits, flonums" ,floats->bits float32 (0.5 -0.0) (#x3f000000 #x80000000))
    ("bits->floats, float32" ,bits->floats float32 (#x3eaaaaab #xc0000000) (0.3333333432674408 -2.0))
    ("bits->floats, float64" ,bits->floats float64 (#x3fd5555555555555) (0.3333333333333333))))

(for ([row (in-list rows)])
  (match-define (list name f dt given expected) row)
  (test-case name
    (check-equal? (f dt given) expected)))
