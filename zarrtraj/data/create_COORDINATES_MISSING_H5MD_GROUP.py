from zarrtraj.tests.datafiles import (
    COORDINATES_SYNTHETIC_H5MD,
    COORDINATES_SYNTHETIC_ZARRMD,
)
from MDAnalysisTests.datafiles import COORDINATES_TOPOLOGY
import zarr
import h5py
import MDAnalysis as mda
import numpy as np


def create_COORDINATES_MISSING_H5MD_GROUP(root):
    del root["h5md"]


def main():
    z = zarr.open_group(COORDINATES_SYNTHETIC_ZARRMD, mode="r")
    outz = zarr.open_group(
        "COORDINATES_MISSING_H5MD_GROUP_ZARRMD.zarrmd", mode="a"
    )
    zarr.convenience.copy_all(z, outz)
    create_COORDINATES_MISSING_H5MD_GROUP(outz)

    h = h5py.File(COORDINATES_SYNTHETIC_H5MD, "r")
    outh = h5py.File("COORDINATES_MISSING_H5MD_GROUP_H5MD.h5md", "a")
    for obj in h.keys():
        h.copy(obj, outh)
    create_COORDINATES_MISSING_H5MD_GROUP(outh)
    h.close()
    outh.close()


if __name__ == "__main__":
    main()
