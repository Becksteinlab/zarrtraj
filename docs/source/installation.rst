Getting Started
===============

Installation
############

With pip
--------

Zarrtraj is available `via pip <https://pypi.org/project/zarrtraj/>`_. To install, simply use::

    pip install zarrtraj

From source with conda
----------------------

Ensure that you have `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ installed.

Clone the repository::

    git clone https://github.com/Becksteinlab/zarrtraj.git .

Create a virtual environment and activate it::

    conda create --name zarrtraj
    conda activate zarrtraj

Build this package from source::

    pip install -e <path/to/repo>

Development installation
------------------------

After creating and activating a conda environment as described, install 
the package with documentation and testing dependencies::

    pip install -e <path/to/repo>[doc, test]

Then, to install the development dependencies::

    conda env update --name zarrtraj --file devtools/conda-envs/test_env.yaml

Or the documentation building dependencies::

    conda env update --name zarrtraj --file docs/requirements.yaml

Or the benchmarking dependencies (this may have to be in a separate conda environment)::

    conda env update --name zarrtraj --file devtools/conda-envs/asv_env.yaml
