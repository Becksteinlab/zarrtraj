"""
zarrtraj
This is a kit that provides the ability to read H5MD trajectory data into MDAnalysis using Zarr
"""

from importlib.metadata import version
from .ZARR import ZARRH5MDReader, ZARRMDWriter


__version__ = version("zarrtraj")
