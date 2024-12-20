from zarrtraj.tests.datafiles import (
    COORDINATES_SYNTHETIC_H5MD,
    COORDINATES_SYNTHETIC_ZARRMD,
)
from MDAnalysisTests.datafiles import COORDINATES_TOPOLOGY
import zarr
import h5py
import MDAnalysis as mda
import numpy as np


def create_COORDINATES_MISSING_TIME_DSET(root):
    del root["particles/trajectory/position/time"]
    del root["particles/trajectory/velocity/time"]
    del root["particles/trajectory/force/time"]


def main():
    z = zarr.open_group(COORDINATES_SYNTHETIC_ZARRMD, mode="r")
    outz = zarr.open_group(
        "COORDINATES_MISSING_TIME_DSET_ZARRMD.zarrmd", mode="a"
    )
    zarr.convenience.copy_all(z, outz)
    create_COORDINATES_MISSING_TIME_DSET(outz)

    h = h5py.File(COORDINATES_SYNTHETIC_H5MD, "r")
    outh = h5py.File("COORDINATES_MISSING_TIME_DSET_H5MD.h5md", "a")
    for obj in h.keys():
        h.copy(obj, outh)
    create_COORDINATES_MISSING_TIME_DSET(outh)
    h.close()
    outh.close()


if __name__ == "__main__":
    main()
