"""
Location of data files
======================

Use as ::

    from zarrtraj.tests.datafiles import *

"""

__all__ = [
    "COORDINATES_ZARRTRAJ",  # traj created with create_zarrtraj_data.py
    "ZARRTRAJ_xvf",  # sample topology
]

from importlib import resources
from pathlib import Path

_data_ref = resources.files('zarrtraj.data')

COORDINATES_ZARRTRAJ = (_data_ref / "test.zarrtraj").as_posix()
ZARRTRAJ_xvf = (_data_ref / "test_topology.pdb").as_posix()

del resources
