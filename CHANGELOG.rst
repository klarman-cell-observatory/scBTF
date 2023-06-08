1.7.1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
July 29, 2022

* Make command-line tool and API work with PegasusIO v0.7.0.
* Rename function ``arcsinh_transform`` to ``arcsinh``, and allow set a user-specified count matrix for the transformation.
* Add functions `log1p <./api/pegasus.log1p.html>`_ and `normalize <./api/pegasus.normalize.html>`_ for preprocessing the count matrix.

1.7.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
July 5, 2022

**New Features**

* Add `pegasus.elbowplot <./api/pegasus.elbowplot.html>`_ function to generate elbowplot, with an automated suggestion on number of PCs to be selected based on random matrix theory (`[Johnstone 2001] <https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-2/On-the-distribution-of-the-largest-eigenvalue-in-principal/10.1214/aos/1009210544.full>`_ and `[Shekhar 2022] <https://elifesciences.org/articles/73809>`_).
* Add `arcsinh_transform <./api/pegasus.arcsinh_transform.html>`_ function for arcsinh transformation on the count matrix.

**Improvement**

* Function ``nearest_neighbors`` has additional argument ``n_comps`` to allow use part of the components from the source embedding for calculating the nearest neighbor graph.
* Add ``n_comps`` argument for ``run_harmony``, ``tsne``, ``umap``, and ``fle`` (argument name is ``rep_ncomps``) functions to allow select part of the components from the source embedding.
* Function ``scatter`` can plot multiple components and bases.