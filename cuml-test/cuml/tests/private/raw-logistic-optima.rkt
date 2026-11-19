#lang racket/base

(module+ main
  (require (only-in racket/port with-output-to-string)
           (only-in racket/pretty pretty-write)
           (only-in racket/string string-trim)
           (only-in racket/system system)
           (only-in "logistic-data.rkt"
                    fit-parts
                    lr-fit-coef
                    lr-fit-intercept
                    lr-fit-n-iter
                    setting-C
                    setting-l1-ratio
                    setting-penalty
                    tight-max-iter
                    tight-tol)
           (only-in "raw-breast-cancer.rkt" raw-breast-cancer)
           (only-in "real-logistic-data.rkt" raw-optima-path raw-optima-settings tight-fit))
  (define described
    (string-trim (with-output-to-string (lambda () (system "git describe --always --dirty")))))
  (define fits
    (for/list ([s (in-list raw-optima-settings)])
      (fit-parts (tight-fit raw-breast-cancer 'float64 s))))
  (for ([s (in-list raw-optima-settings)]
        [f (in-list fits)])
    (printf "~a: n-iter ~a\n" s (lr-fit-n-iter f)))
  (define entries
    (for/list ([s (in-list raw-optima-settings)]
               [f (in-list fits)])
      (list (list (setting-penalty s) (setting-C s) (setting-l1-ratio s))
            (lr-fit-coef f)
            (lr-fit-intercept f))))
  (with-output-to-file
   raw-optima-path
   #:exists 'truncate
   (lambda ()
     (for-each displayln
               (list ";; Raw breast-cancer's float64 logistic optima, one per capped setting."
                     ";; Written by `racket private/raw-logistic-optima.rkt` on a GPU host:"
                     ";; cuML's qn solver through logistic-regression-fit, dtype float64,"
                     (format ";; tol ~a, max-iter ~a," (tight-tol 'float64) tight-max-iter)
                     (format ";; at ~a (git describe --always --dirty)." described)
                     ";; Each entry: ((penalty C l1-ratio) coef-rows intercepts)."))
     (pretty-write entries))))
