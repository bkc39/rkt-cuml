#lang info

#|review: ignore|#

(define test-omit-paths '("python"))
(define test-timeouts
  '(("memory-test.rkt" 180) ("twin-test.rkt" 180)
                            ("kmeans-memory-test.rkt" 180)
                            ("kmeans-estimator-memory-test.rkt" 180)
                            ("kmeans-twin-test.rkt" 180)))
