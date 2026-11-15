#lang racket/base

(require (only-in cuml/linear-model mbsgd-classifier-predict)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in raft device-vector->list)
         (only-in threading ~>)
         (only-in "private/linear-data.rkt" dtypes on-matrix)
         (only-in "private/mbsgd-data.rkt" blob-rows learning-rates normal-rows penalties reg-target)
         (only-in "private/mbsgd-twin-data.rkt"
                  clf-fit
                  clf-result
                  mbsgd-twin-results
                  reg-fit
                  their-parts
                  twin-request)
         (only-in "private/python-env.rkt" test-twin))

(define rows (normal-rows 300 5 21))
(define targets (reg-target rows 21))
(define-values (blobs labels) (blob-rows 300 4 22 #:minority 0.3))

(define base (hash '#:batch-size 16 '#:eta0 0.01 '#:epochs 50))

(define (check-twin grid request check)
  (for ([g (in-list grid)]
        [theirs (in-list (mbsgd-twin-results (map (lambda (row) (apply request row)) grid)))])
    (apply check theirs g)))

(test-twin "the regressor is Python cuML's MBSGDRegressor bit for bit"
  (check-twin (for*/list ([dt (in-list dtypes)]
                          [fi (in-list '(#t #f))]
                          [lr (in-list learning-rates)]
                          [penalty (in-list penalties)])
                (list dt (hash-set* base '#:fit-intercept fi '#:learning-rate lr '#:penalty penalty)))
              (lambda (dt s) (twin-request "cuml-reg" dt rows s #:y targets))
              (lambda (theirs dt s)
                (test-case (format "~a ~a" dt s)
                  (check-equal? (reg-fit dt rows targets s) (their-parts theirs dt))))))

(define (clf-predict dt s loss)
  (~> (clf-result dt blobs labels loss s)
      (mbsgd-classifier-predict (on-matrix blobs dt))
      device-vector->list))

(test-twin "squared-loss and intercept-free log classifiers are Python cuML's MBSGDClassifier"
  (check-twin (append (for*/list ([dt (in-list dtypes)]
                                  [fi (in-list '(#t #f))])
                        (list dt 'squared-loss (hash-set* base '#:fit-intercept fi)))
                      (for/list ([dt (in-list dtypes)])
                        (list dt 'log (hash-set* base '#:fit-intercept #f))))
              (lambda (dt loss s) (twin-request "cuml-clf" dt blobs s #:codes labels #:loss loss))
              (lambda (theirs dt loss s)
                (test-case (format "~a ~a ~a" dt loss s)
                  (check-equal? (clf-fit dt blobs labels loss s) (their-parts theirs dt) "fit")
                  (when (eq? loss 'squared-loss)
                    (check-equal? (clf-predict dt s loss) (hash-ref theirs 'predict) "predict"))))))

(test-twin "corrected hinge and log fits are fit_sgd on the ones column and the -1/+1 target"
  (check-twin (append (for*/list ([dt (in-list dtypes)]
                                  [fi (in-list '(#t #f))])
                        (list dt 'hinge (hash-set* base '#:fit-intercept fi)))
                      (for/list ([dt (in-list dtypes)])
                        (list dt 'log (hash-set* base '#:fit-intercept #t))))
              (lambda (dt loss s) (twin-request "fit-sgd" dt blobs s #:codes labels #:loss loss))
              (lambda (theirs dt loss s)
                (test-case (format "~a ~a ~a" dt loss s)
                  (match-define (list coef intercept) (clf-fit dt blobs labels loss s))
                  (match-define (and parts (list their-coef _)) (their-parts theirs dt))
                  (if (hash-ref s '#:fit-intercept)
                      (check-equal? (append coef (list intercept)) their-coef)
                      (check-equal? (list coef intercept) parts))))))

(test-twin "a seeded shuffle is MBSGDRegressor on the rows in numpy's mt19937 Fisher-Yates order"
  (check-twin (for*/list ([dt (in-list dtypes)]
                          [seed (in-list '(7 4294967295))])
                (list dt seed))
              (lambda (dt seed) (twin-request "permute" dt rows base #:y targets #:seed seed))
              (lambda (theirs dt seed)
                (test-case (format "~a seed ~a" dt seed)
                  (check-equal? (reg-fit dt rows targets base #:random-state seed)
                                (their-parts theirs dt))))))
