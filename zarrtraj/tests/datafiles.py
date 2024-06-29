"""
Location of data files
======================

Use as ::

    from zarrtraj.tests.datafiles import *

"""

__all__ = [
    "COORDINATES_SYNTHETIC_H5MD",
    "COORDINATES_SYNTHETIC_ZARRMD",
    "COORDINATES_MISSING_H5MD_GROUP_H5MD",
    "COORDINATES_MISSING_H5MD_GROUP_ZARRMD",
    "COORDINATES_MISSING_TIME_DSET_H5MD",
    "COORDINATES_MISSING_TIME_DSET_ZARRMD",
    "COORDINATES_VARIED_STEPS_H5MD",
    "COORDINATES_VARIED_STEPS_ZARRMD",
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
COORDINATES_MISSING_H5MD_GROUP_H5MD = (
    _data_ref / "COORDINATES_MISSING_H5MD_GROUP_H5MD.h5md"
).as_posix()
COORDINATES_MISSING_H5MD_GROUP_ZARRMD = (
    _data_ref / "COORDINATES_MISSING_H5MD_GROUP_ZARRMD.zarrmd"
).as_posix()
COORDINATES_MISSING_TIME_DSET_H5MD = (
    _data_ref / "COORDINATES_MISSING_TIME_DSET_H5MD.h5md"
).as_posix()
COORDINATES_MISSING_TIME_DSET_ZARRMD = (
    _data_ref / "COORDINATES_MISSING_TIME_DSET_ZARRMD.zarrmd"
).as_posix()
COORDINATES_VARIED_STEPS_H5MD = (
    _data_ref / "COORDINATES_VARIED_STEPS_H5MD.h5md"
).as_posix()
COORDINATES_VARIED_STEPS_ZARRMD = (
    _data_ref / "COORDINATES_VARIED_STEPS_ZARRMD.zarrmd"
).as_posix()

del resources
