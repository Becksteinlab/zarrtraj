"""
Location of data files
======================

Use as ::

    from zarrtraj.data.files import *

"""

__all__ = [
    "COORDINATES_ZARRTRAJ",  # traj created with create_zarrtraj_data.py
    "ZARRTRAJ_xvf",  # sample topology
]

from pkg_resources import resource_filename

COORDINATES_ZARRTRAJ = resource_filename(__name__, "test.zarrtraj")
ZARRTRAJ_xvf = resource_filename(__name__, "test_topology.pdb")