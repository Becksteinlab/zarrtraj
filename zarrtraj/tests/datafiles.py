"""
Location of data files
======================

Use as ::

    from zarrtraj.tests.datafiles import *

"""

__all__ = [
    "COORDINATES_SYNTHETIC_H5MD",
    "COORDINATES_SYNTHETIC_ZARRMD",
]

from importlib import resources
from pathlib import Path

_data_ref = resources.files("zarrtraj.data")

COORDINATES_SYNTHETIC_H5MD = (
    _data_ref / "COORDINATES_SYNTHETIC_H5MD.h5md"
).as_posix()
COORDINATES_SYNTHETIC_ZARRMD = (
    _data_ref / "COORDINATES_SYNTHETIC_ZARRMD.zarrmd"
).as_posix()

del resources
