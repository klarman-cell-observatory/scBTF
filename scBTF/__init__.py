""" Bayesian Tensor Factorization of single cell rnaseq data  """

from .sc_tensor import SingleCellTensor
from .sc_factors import FactorizationSet, Factorization
from scBTF.sc_btf import SingleCellBTF
from .bayesian_parafac import BayesianCP


try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # < Python 3.8: Use backport module
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version('scbtf')
    del version
except PackageNotFoundError:
    pass