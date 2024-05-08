"""
Location of data files
======================

Use as ::

    from zarrtraj.tests.datafiles import *

"""

__all__ = [
    "COORDINATES_ZARRTRAJ",  # synthetic traj from create_zarrtraj_data.py
    "ZARRTRAJ_xvf",  # real traj created with create_ZARRTRAJ_xvf.py
]

from importlib import resources
from pathlib import Path

_data_ref = resources.files('zarrtraj.data')

COORDINATES_ZARRTRAJ = (_data_ref / "test.zarrtraj").as_posix()
ZARRTRAJ_xvf = (_data_ref / "cobrotoxin.zarrtraj").as_posix()

del resources
