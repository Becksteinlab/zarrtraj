"""
zarrtraj
This is a kit that provides the ability to read and write trajectory data in the Zarr file format
"""

# Add imports here
from importlib.metadata import version
from .ZARRTRAJ import *
from .ZARR import *
from .cache import *

from MDAnalysis.lib import util
import numpy as np
import zarr


__version__ = version("zarrtraj")
