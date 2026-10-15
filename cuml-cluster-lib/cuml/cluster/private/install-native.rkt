#lang racket/base

(require (only-in cuml/core/private/install-native native-pre-installer))

(provide pre-installer)

(define pre-installer (native-pre-installer "cluster"))
