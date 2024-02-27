"""
zarrtraj
This is a kit that provides the ability to read and write trajectory data in the Zarr file format
"""

# Add imports here
from importlib.metadata import version
from .ZARRTRAJ import *

# Monkey patch to prevent mda from selecting
# ChainReader for zarr group
from MDAnalysis import _READER_HINTS
from MDAnalysis.lib import util
import numpy as np
import zarr

print("importing zarrtraj...")

@staticmethod
def _format_hint(thing):
    return (not isinstance(thing, np.ndarray) and
            not isinstance(thing, zarr.Group) and
            util.iterable(thing) and
            not util.isstream(thing))

_READER_HINTS["CHAIN"] = _format_hint


__version__ = version("zarrtraj")
