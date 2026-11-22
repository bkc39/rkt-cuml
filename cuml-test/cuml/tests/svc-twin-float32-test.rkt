#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/svm-data.rkt" kernels)
         (only-in "private/svm-twin-data.rkt" check-cuml-twins trial twin-fixtures twin-gammas))

(test-twin "T1, T2: float32 fits are Python cuML's, bit for bit"
  (check-cuml-twins (for*/list ([fx (in-list (twin-fixtures))]
                                [kernel (in-list kernels)]
                                [gamma (in-list twin-gammas)])
                      (trial fx 'float32 kernel gamma))))
