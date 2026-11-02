#lang racket/base

(require (only-in cuml/scribblings/guide/helpers dataset-column dataset-columns features sizes)
         (only-in datasets/core dataset-table-names load-dataset-table)
         (only-in pkg/lib pkg-directory)
         (only-in racket/list append-map last make-list remove-duplicates second)
         (only-in racket/match match match-define)
         (only-in racket/path file-name-from-path)
         (only-in racket/string string-contains?)
         (only-in rackunit check-equal? check-not-false check-true test-case)
         (only-in setup/getinfo get-info/full)
         (only-in threading ~>)
         (only-in "private/guide-data.rkt" chapter-file flattened-text read-document))

(define (included-sections datum)
  (match datum
    [(list 'include-section (? string? section)) (list section)]
    [(cons head tail) (append (included-sections head) (included-sections tail))]
    [_ '()]))

(define (guide-chapters [file (collection-file-path "guide.scrbl" "cuml" "scribblings")])
  (define-values (dir _name _directory?) (split-path file))
  (append-map (lambda (section)
                (define chapter (build-path dir section))
                (cons chapter (guide-chapters chapter)))
              (~> file read-document included-sections)))

(define dataset-citations
  '((iris datasets-core ("doi:10.24432/C56C76" "CC BY 4.0") ("doi:10.24432/C56C76. CC BY 4.0"))
    (wine datasets-core ("doi:10.24432/C5PC7J" "CC BY 4.0") ("doi:10.24432/C5PC7J. CC BY 4.0"))
    (faithful datasets-core
              ("R 4.3.3" "GPL-2 or GPL-3")
              ("R 4.3.3 datasets: GPL-2.0-only OR GPL-3.0-only"))
    (breast-cancer datasets-core
                   ("doi:10.24432/C5DW2B" "CC BY 4.0")
                   ("doi:10.24432/C5DW2B. CC BY 4.0"))
    (digits datasets-uci
            ("doi:10.24432/C50P49" "CC BY 4.0")
            ("from the UCI Machine Learning Repository under CC BY 4.0"
             "Optical Recognition of Handwritten Digits, doi:10.24432/C50P49."))))

(define dataset-use #px"\\((?:load-dataset-table|dataset-columns?|features) '([a-z-]+)")

(define (chapter-datasets chapter)
  (map string->symbol (regexp-match* dataset-use (flattened-text chapter) #:match-select cadr)))

(define (notice-of pkg)
  (define dir (pkg-directory (symbol->string pkg)))
  (check-not-false dir (format "pkg-directory: ~a is not installed" pkg))
  (flattened-text (build-path dir "NOTICE")))

(define (dataset-introductions)
  (for*/fold ([introductions '()]
              #:result (reverse introductions))
             ([chapter (in-list (guide-chapters))]
              [dataset (in-list (remove-duplicates (chapter-datasets chapter)))]
              #:unless (assq dataset introductions))
    (cons (cons dataset chapter) introductions)))

(test-case "the guide chapter that introduces a dataset cites its DOI or licence, as its package's NOTICE gives them"
  (define introductions (dataset-introductions))
  (check-equal? (map car introductions) (map car dataset-citations))
  (for ([citation (in-list dataset-citations)])
    (match-define (list dataset pkg cited notice-texts) citation)
    (define chapter (cdr (assq dataset introductions)))
    (for ([text (in-list notice-texts)])
      (check-not-false (string-contains? (notice-of pkg) text) (format "~a NOTICE: ~a" pkg text)))
    (for ([text (in-list cited)])
      (check-not-false (string-contains? (flattened-text chapter) text)
                       (format "~a: ~a" chapter text)))))

(test-case "digits is introduced last, by the Embeddings chapter"
  (check-equal? (~> (dataset-introductions) last cdr file-name-from-path)
                (string->path "embeddings.scrbl")))

(define (install-line)
  (match-define (list line)
    (regexp-match #px"@commandline\\{raco pkg install [^}]*\\}"
                  (flattened-text (chapter-file "getting-started.scrbl"))))
  line)

(define (examples-intro)
  (define text (flattened-text (chapter-file "examples.scrbl")))
  (substring text 0 (caar (regexp-match-positions #rx"@page-helpers-block" text))))

(define (umbrella-dependencies)
  (for/list ([dependency (in-list ((get-info/full (pkg-directory "datasets")) 'deps))])
    (match dependency
      [(cons name _) name]
      [name name])))

(test-case "the install line and the Examples intro name the datasets umbrella, which carries every cited package"
  (check-not-false (regexp-match? #px" datasets( |\\})" (install-line)) "install line: datasets")
  (check-not-false (regexp-match? #px"@tt\\{datasets\\}" (examples-intro)) "Examples intro: datasets")
  (for ([pkg (in-list (remove-duplicates (map second dataset-citations)))])
    (check-not-false (member (symbol->string pkg) (cons "datasets-core" (umbrella-dependencies)))
                     (format "the datasets umbrella carries ~a" pkg))))

(test-case "digits loads without a GPU: 1797 rows of 64 pixels from 0 to 16 and the digit"
  (define digits (load-dataset-table 'digits))
  (define names (dataset-table-names digits))
  (define pixels (features 'digits "digit"))
  (check-equal? (length names) 65)
  (check-equal? (last names) "digit")
  (check-equal? (sizes (dataset-column 'digits "digit")) '(183 182 182 181 181 180 179 178 177 174))
  (define columns (dataset-columns 'digits pixels))
  (check-equal? (map length columns) (make-list 64 1797))
  (check-true (for*/and ([column (in-list columns)]
                         [pixel (in-list column)])
                (<= 0 pixel 16))))
