#lang racket/base

(require (only-in racket/match match-define struct*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/svm-data.rkt" kernels svm-case)
         (only-in "private/svm-oracle.rkt" check-kkt)
         (only-in "private/svm-twin-data.rkt" breast-cancer check-cuml-twins fixture trial twin-gammas))

(define tumours (breast-cancer))

(test-gpu "standardised breast cancer: the KKT law holds for rbf and linear"
  (match-define (struct* fixture ([rows rows] [labels labels])) tumours)
  (for* ([dt (in-list dtypes)]
         [kernel (in-list '(linear rbf))])
    (check-kkt (svm-case (format "breast cancer ~a ~a" dt kernel) rows labels #f 1 kernel dt 'scale))))

(test-twin "standardised breast cancer: T1, T2 against Python cuML"
  (check-cuml-twins (for*/list ([dt (in-list dtypes)]
                                [kernel (in-list kernels)]
                                [gamma (in-list twin-gammas)])
                      (trial tumours dt kernel gamma))))
