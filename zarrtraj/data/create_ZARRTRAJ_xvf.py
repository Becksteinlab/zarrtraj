"""Used to write ZARRTRAJ_xvf for testing with
real trajectory"""

import zarrtraj
from MDAnalysisTests.datafiles import TPR_xvf, TRR_xvf
import zarr
import MDAnalysis as mda

u = mda.Universe(TPR_xvf, TRR_xvf)
z = zarr.open_group("cobrotoxin.zarrtraj", mode="a")
with mda.Writer(
    z,
    n_atoms=u.trajectory.n_atoms,
    n_frames=u.trajectory.n_frames,
    format="ZARRTRAJ",
) as w:
    for ts in u.trajectory:
        w.write(u)
