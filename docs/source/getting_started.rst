Getting Started
===============

Installation
############

To build zarrtraj from source, we highly recommend using virtual environments.
If possible, we strongly recommend that you use
`Anaconda <https://docs.conda.io/en/latest/>`_ as your package manager.
Below we provide instructions both for installing into a `conda` environment.

With conda
----------

Ensure that you have `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ installed.

Create a virtual environment and activate it::

    conda create --name zarrtraj
    conda activate zarrtraj

Build this package from source::

    pip install -e .

Development environment installation
------------------------------------

Perform a normal conda installation as described, and then 
install the development and documentation dependencies::

    conda env update --name zarrtraj --file devtools/conda-envs/test_env.yaml
    conda env update --name zarrtraj --file docs/requirements.yaml