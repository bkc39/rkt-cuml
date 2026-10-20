#lang scribble/manual
@title{cuML: GPU machine learning for Racket}
@author{bkc}

This library binds @hyperlink["https://github.com/rapidsai/cuml"]{NVIDIA
cuML}, the RAPIDS library of machine-learning algorithms on the GPU: k-means,
DBSCAN, single-linkage and spectral clustering, with a scikit-learn-style
estimator interface. It works
on the device arrays and device resources of @racketmodname[raft].

@bold{License.} This package is distributed under @bold{Apache-2.0}.

@bold{Acknowledgements.} cuML and RAFT, which do all of this package's work,
are developed by NVIDIA's RAPIDS cuML and RAFT teams.

@bold{AI Disclosure.} This package and its documentation were created with the
use of AI tools.

Install it with @exec{raco pkg install cuml}.

@local-table-of-contents[]

@include-section["reference.scrbl"]
