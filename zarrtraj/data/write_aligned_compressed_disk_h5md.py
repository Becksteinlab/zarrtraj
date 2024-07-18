import zarrtraj
import MDAnalysis as mda

# This requires MDAnalysis >= 2.8.0

u = mda.Universe(
    "zarrtraj/data/yiip_equilibrium/YiiP_system.pdb",
    "zarrtraj/data/yiip_equilibrium/YiiP_system_90ns_center_aligned.xtc",
)

with mda.Writer(
    "zarrtraj/data/yiip_aligned_compressed.h5md",
    n_atoms=u.trajectory.n_atoms,
    n_frames=u.trajectory.n_frames,
    compression="gzip",
    compression_opts=9,
    chunks=(9, u.trajectory.n_atoms, 3),
) as W:
    for ts in u.trajectory:
        W.write(u.atoms)
