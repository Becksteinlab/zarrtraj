from MDAnalysisTests.datafiles import COORDINATES_TOPOLOGY
from zarrtraj.tests.datafiles import (
    COORDINATES_SYNTHETIC_H5MD,
    COORDINATES_SYNTHETIC_ZARRMD,
)
import zarr
import h5py
import MDAnalysis as mda
import numpy as np


def create_COORDINATES_VARIED_STEPS(root):
    """
    1. Give positions and box/edges fixed time steps
    2. Create four time-dependent observables:
    - trajectory/obsv1 with fixed time steps
    - obsv1 with fixed time steps
    - trajectory/obsv2 with explicit time steps
    - obsv2 with explicit time steps
    3. Create two time-independent observables
    - obsv3
    - trajectory/obsv3
    4. Modify velocity to be sampled at a different rate than position and
       use explicit time steps
    5. Modify force to be sampled at a different rate than position and
       use fixed time steps
    """
    del root["particles/trajectory/box/edges/step"]
    del root["particles/trajectory/box/edges/time"]
    del root["particles/trajectory/position/step"]
    del root["particles/trajectory/position/time"]
    root["particles/trajectory/box/edges/step"] = 1
    root["particles/trajectory/box/edges/time"] = 1.0
    root["particles/trajectory/box/edges/step"].attrs["offset"] = 0
    root["particles/trajectory/box/edges/time"].attrs["offset"] = 0.0
    root["particles/trajectory/box/edges/time"].attrs["unit"] = "ps"

    root["particles/trajectory/position/step"] = root[
        "particles/trajectory/box/edges/step"
    ]
    root["particles/trajectory/position/time"] = root[
        "particles/trajectory/box/edges/time"
    ]
    # Zarr doesn't copy attributes on assignment, so set offset and unit again
    root["particles/trajectory/position/step"].attrs["offset"] = 0
    root["particles/trajectory/position/time"].attrs["offset"] = 0.0
    root["particles/trajectory/position/time"].attrs["unit"] = "ps"

    root["observables/trajectory/obsv1/step"] = 2
    root["observables/trajectory/obsv1/time"] = 2.0
    root["observables/trajectory/obsv1/step"].attrs["offset"] = 1
    root["observables/trajectory/obsv1/time"].attrs["offset"] = 1.0
    root["observables/trajectory/obsv1/time"].attrs["unit"] = "ps"
    root["observables/trajectory/obsv1/value"] = np.array([4, 6, 8])

    root["observables/obsv1/step"] = 2
    root["observables/obsv1/time"] = 2.0
    root["observables/obsv1/step"].attrs["offset"] = 1
    root["observables/obsv1/time"].attrs["offset"] = 1.0
    root["observables/obsv1/time"].attrs["unit"] = "ps"
    root["observables/obsv1/value"] = np.array([4, 6, 8])

    root["observables/trajectory/obsv2/step"] = np.array([1, 3, 5])
    root["observables/trajectory/obsv2/time"] = np.array([1.0, 3.0, 5.0])
    root["observables/trajectory/obsv2/time"].attrs["unit"] = "ps"
    root["observables/trajectory/obsv2/value"] = np.array([4, 6, 8])

    root["observables/obsv2/step"] = np.array([1, 3, 5])
    root["observables/obsv2/time"] = np.array([1.0, 3.0, 5.0])
    root["observables/obsv2/time"].attrs["unit"] = "ps"
    root["observables/obsv2/value"] = np.array([4, 6, 8])

    root["observables/obsv3/value"] = np.array([4, 6, 8])
    root["observables/trajectory/obsv3/value"] = np.array([4, 6, 8])

    vel_val = root["particles/trajectory/velocity/value"][:]
    del root["particles/trajectory/velocity/step"]
    del root["particles/trajectory/velocity/time"]
    del root["particles/trajectory/velocity/value"]
    root["particles/trajectory/velocity/step"] = np.array([0, 1, 2])
    root["particles/trajectory/velocity/time"] = np.array([0, 1.0, 2.0])
    root["particles/trajectory/velocity/time"].attrs["unit"] = "ps"
    root["particles/trajectory/velocity/value"] = vel_val[:3]
    root["particles/trajectory/velocity/value"].attrs["unit"] = "Angstrom ps-1"

    force_val = root["particles/trajectory/force/value"][:]
    del root["particles/trajectory/force/step"]
    del root["particles/trajectory/force/time"]
    del root["particles/trajectory/force/value"]
    root["particles/trajectory/force/step"] = 1
    root["particles/trajectory/force/step"].attrs["offset"] = 1
    root["particles/trajectory/force/time"] = 1.0
    root["particles/trajectory/force/time"].attrs["offset"] = 1.0
    root["particles/trajectory/force/time"].attrs["unit"] = "ps"
    root["particles/trajectory/force/value"] = force_val[1:]
    root["particles/trajectory/force/value"].attrs[
        "unit"
    ] = "kJ mol-1 Angstrom-1"


def main():
    z = zarr.open_group(COORDINATES_SYNTHETIC_ZARRMD, mode="r")
    outz = zarr.open_group("COORDINATES_VARIED_STEPS_ZARRMD.zarrmd", mode="a")
    zarr.convenience.copy_all(z, outz)
    create_COORDINATES_VARIED_STEPS(outz)

    h = h5py.File(COORDINATES_SYNTHETIC_H5MD, "r")
    outh = h5py.File("COORDINATES_VARIED_STEPS_H5MD.h5md", "a")
    for obj in h.keys():
        h.copy(obj, outh)
    create_COORDINATES_VARIED_STEPS(outh)
    h.close()
    outh.close()


if __name__ == "__main__":
    main()
