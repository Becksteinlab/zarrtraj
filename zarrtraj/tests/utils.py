"""Buffer-related helper functions for testing."""

import numpy as np
from MDAnalysis.analysis import distances
import socket


# Helper Functions
def get_memory_usage(writer):
    mem = (
        writer._time_buffer.nbytes
        + writer._step_buffer.nbytes
        + writer._dimensions_buffer.nbytes
        + writer._pos_buffer.nbytes
        + writer._force_buffer.nbytes
        + writer._vel_buffer.nbytes
    )
    for key in writer._obsv_buffer:
        mem += writer._obsv_buffer[key].nbytes
    return mem


def get_frame_size(universe):
    has = []
    data_blacklist = ["step", "time", "dt"]
    ts = universe.trajectory[0]
    mem_per_frame = 0
    try:
        has.append(ts.data["step"])
    except KeyError:
        has.append(ts.frame)
    has.append(ts.time)
    if ts.dimensions is not None:
        has.append(ts.triclinic_dimensions)
    if ts.has_positions:
        has.append(ts.positions)
    if ts.has_velocities:
        has.append(ts.velocities)
    if ts.has_forces:
        has.append(ts.forces)
    for key in ts.data:
        if key not in data_blacklist:
            has.append(ts.data[key])
    for dataset in has:
        mem_per_frame += dataset.size * dataset.itemsize
    return mem_per_frame


def get_n_closest_water_molecules(prot_ag, wat_ag, n):
    # returns a numpy array of the indices of the n closest water molecules
    # to a protein across all frames of a trajectory
    if n > wat_ag.n_atoms:
        raise ValueError("n must be less than the number of water molecules")

    result = np.empty((prot_ag.universe.trajectory.n_frames, n), dtype=int)

    i = 0
    for ts in prot_ag.universe.trajectory:
        dist = distances.distance_array(
            prot_ag.positions, wat_ag.positions, box=prot_ag.dimensions
        )

        minvals = np.empty(wat_ag.n_atoms)
        for j in range(wat_ag.n_atoms):
            minvals[j] = np.min(dist[:, j])

        result[i] = np.argsort(minvals)[:n]
        i += 1

    return result


def find_free_port():
    """Find a free port on the host machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
