from setuptools import setup, find_packages
from codecs import open
from pathlib import Path
from os import path


here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="scBTF",
    version='0.1.6',
    description="scBTF is a Python package for Bayesian Tensor Factorization of single cell RNA-seq data",
    long_description=long_description,
    url="https://github.com/dan-broad/scBTF",
    author="Daniel Chafamo",
    author_email="chafamodaniel@gmail.com",
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="single cell/nucleus genomics analysis",
    packages=find_packages(),
    install_requires=[
        'adpbulk', 'anndata>=0.7.1', 'gseapy>=1.0.2', 'matplotlib>=2.0.0', 'numpy', 'pandas>=1.2.0',
        'pyro-ppl>=1.8.3', 'torch>=1.13.1', 'rich>=13.0.0', 'scikit-learn>=1.2.0', 'scipy', 'seaborn',
        'setuptools', 'statannotations>=0.5.0', 'tensorly>=0.7.0', 'tqdm'
    ],
    python_requires="~=3.7",
    package_data={
        "scBTF": ["resources/gene_sets/hgnc_complete_set_2020-07-01.txt"],
    }
)