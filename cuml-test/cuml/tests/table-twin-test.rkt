#lang racket/base

(require (only-in cuml table->device-matrix)
         (only-in racket/match match-define)
         (only-in racket/math nan? infinite?)
         racket/runtime-path
         (only-in rackunit check-equal?)
         (only-in raft device-matrix->list*)
         (only-in "private/python-env.rkt" run-twin test-twin)
         (only-in "private/table-data.rkt"
                  edge-values
                  edge-values32
                  host-rows
                  numeric-selections
                  selection
                  synthetic-table))

(define-runtime-path twin "python/table_twin.py")

(struct twin-case (name table columns dtype))

(define (json-safe? x)
  (not (or (nan? x) (infinite? x) (eqv? x -0.0))))

(define (json-table xs)
  (synthetic-table '("x") (list (filter json-safe? xs))))

(define (cases)
  (append (for*/list ([s (in-list (numeric-selections))]
                      [element-type (in-list '(float32 float64))])
            (match-define (selection name t columns) s)
            (twin-case (format "~a ~a" name element-type) t columns element-type))
          (list (twin-case "edge float32" (json-table edge-values32) '("x") 'float32)
                (twin-case "edge float64" (json-table edge-values) '("x") 'float64))))

(define (twin-case->jsexpr c)
  (match-define (twin-case name t columns element-type) c)
  (hasheq 'name name
          'names columns
          'columns (apply map list (host-rows t columns))
          'dtype (symbol->string element-type)))

(test-twin "L12: every numeric selection and the JSON-safe edge values match pandas through cupy"
  (define all (cases))
  (define theirs (hash-ref (run-twin twin (hasheq 'cases (map twin-case->jsexpr all))) 'cases))
  (for ([c (in-list all)]
        [their (in-list theirs)])
    (match-define (twin-case name t columns element-type) c)
    (check-equal?
     (device-matrix->list* (table->device-matrix t #:columns columns #:dtype element-type))
     (hash-ref their 'rows)
     name)))
