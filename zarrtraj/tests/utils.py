"""Buffer-related helper functions for testing."""

import numpy as np


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
