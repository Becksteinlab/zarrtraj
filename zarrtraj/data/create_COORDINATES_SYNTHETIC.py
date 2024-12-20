"""Used to create synthetic test trajectory"""

import zarrtraj
from MDAnalysisTests.datafiles import COORDINATES_TOPOLOGY
import zarr
import h5py
import MDAnalysis as mda
import numpy as np


def create_COORDINATES_SYNTHETIC(uni, root):
    n_atoms = uni.atoms.n_atoms
    h5md = root.create_group("h5md")
    h5md.attrs["version"] = [1, 1]
    author = h5md.create_group("author")
    author.attrs["name"] = "John Doe"
    creator = h5md.create_group("creator")
    creator.attrs["name"] = "MDAnalysis"
    creator.attrs["version"] = mda.__version__

    pgroup = root.create_group("particles/trajectory")

    step_data = np.arange(5, dtype=np.int32)
    time_data = np.arange(5, dtype=np.float32)

    box = pgroup.create_group("box")
    box.attrs["dimension"] = 3
    box.attrs["boundary"] = ["periodic"] * 3

    dim = box.create_group("edges")
    edges_data = np.tile(
        np.array([81.1, 82.2, 83.3, 75, 80, 85], dtype=np.float32), (5, 1)
    )
    edges_data[:, :3] += step_data[:, None]
    edges_data[:, 3:] += step_data[:, None] * 0.1
    # Set each index to edges data at the first dimension transformed
    # using triclinic_vectoors
    triclinic_edges_data = np.empty((5, 3, 3), dtype=np.float32)
    for i in range(5):
        triclinic_edges_data[i] = mda.lib.mdamath.triclinic_vectors(
            edges_data[i]
        )
    dim.create_dataset("value", data=triclinic_edges_data)
    dim["value"].attrs["unit"] = "Angstrom"
    dim.create_dataset("step", data=step_data)
    dim.create_dataset("time", data=time_data)
    dim["time"].attrs["unit"] = "ps"

    pos = pgroup.create_group("position")
    exp = np.logspace(0, 4, base=2, num=5, dtype=np.float32)
    pos_data = (
        np.tile(
            np.arange(3 * n_atoms, dtype=np.float32).reshape(n_atoms, 3),
            (5, 1, 1),
        )
        * exp[:, None, None]
    )
    pos.create_dataset("value", data=pos_data)
    pos["value"].attrs["unit"] = "Angstrom"
    # in zarr, this will copy the array
    # in hdf5, this creates a hard link (as dictated by the H5MD standard)
    pos["step"] = dim["step"]
    pos["time"] = dim["time"]
    pos["time"].attrs["unit"] = "ps"

    vel = pgroup.create_group("velocity")
    vel_data = pos_data / 10
    vel.create_dataset("value", data=vel_data)
    vel["value"].attrs["unit"] = "Angstrom ps-1"
    vel.create_dataset("step", data=step_data)
    vel.create_dataset("time", data=time_data)
    vel["time"].attrs["unit"] = "ps"

    force = pgroup.create_group("force")
    force_data = pos_data / 100
    force.create_dataset("value", data=force_data)
    force["value"].attrs["unit"] = "kJ mol-1 Angstrom-1"
    force.create_dataset("step", data=step_data)
    force.create_dataset("time", data=time_data)
    force["time"].attrs["unit"] = "ps"

    occupancy = root.create_group("observables/occupancy")
    occ_data = np.full((5, 5), 1.0, dtype=np.float64)
    occupancy.create_dataset("value", data=occ_data)


def main():
    u = mda.Universe(COORDINATES_TOPOLOGY)
    z = zarr.open_group("COORDINATES_SYNTHETIC_ZARRMD.zarrmd", mode="w")
    create_COORDINATES_SYNTHETIC(u, z)

    h = h5py.File("COORDINATES_SYNTHETIC_H5MD.h5md", "w")
    create_COORDINATES_SYNTHETIC(u, h)
    h.close()


if __name__ == "__main__":
    main()
