Installation
------------

Pegasus works with Python 3.7, 3.8 and 3.9.

Linux
^^^^^

Ubuntu/Debian
###############

Prerequisites
+++++++++++++++

On Ubuntu/Debian Linux, first install the following dependency by::

	sudo apt install build-essential

Next, you can install Pegasus system-wide by PyPI (see `Ubuntu/Debian install via PyPI`_), or within a Miniconda environment

OpenJDK which is included in Ubuntu official repository::

	sudo apt install default-jdk

Ubuntu/Debian install via PyPI
+++++++++++++++++++++++++++++++++

First, install Python 3, *pip* tool for Python 3 and *Cython* package::

	sudo apt install python3 python3-pip
	python3 -m pip install --upgrade pip
	python3 -m pip install cython

Now install Pegasus with the required dependencies via *pip*::

	python3 -m pip install pegasuspy

or install Pegasus with all dependencies::

	python3 -m pip install pegasuspy[all]

Alternatively, you can install Pegasus with some of the additional optional dependencies as below:

