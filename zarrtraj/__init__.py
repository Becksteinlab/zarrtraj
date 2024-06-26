"""
zarrtraj
This is a kit that provides the ability to read and write trajectory data in the Zarr file format
"""

# Add imports here
from importlib.metadata import version
from .ZARR import ZARRH5MDReader


__version__ = version("zarrtraj")
