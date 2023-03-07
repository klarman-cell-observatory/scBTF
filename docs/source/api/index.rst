.. automodule:: scBTF
    :noindex:

API
===

*scBTF* can also be used as a python package. Import scBTF by::

    from scBTF import SingleCellTensor, SingleCellBTF, FactorizationSet

Tensor Formation
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: .

    SingleCellTensor.from_anndata
    SingleCellTensor.from_anndata_with_regions
    SingleCellTensor.from_anndata_ligand_receptor

Bayesian Tensor Factorization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: .

    BayesianCP.fit
    BayesianCP.precis

Single Cell Tensor Factorize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: .

    SingleCellBTF.factorize
    SingleCellBTF.factorize_hals


Downstream analysis and visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: .

    FactorizationSet.variance_explained
    FactorizationSet.variance_explained_elbow_plot
    FactorizationSet.rank_metrics_plot
    FactorizationSet.plot_components