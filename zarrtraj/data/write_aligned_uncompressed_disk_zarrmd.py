import zarrtraj
import MDAnalysis as mda
import numcodecs

u = mda.Universe(
    "zarrtraj/data/yiip_equilibrium/YiiP_system.pdb",
    "zarrtraj/data/yiip_equilibrium/YiiP_system_90ns_center_aligned.xtc",
)

with mda.Writer(
    "zarrtraj/data/yiip_aligned_uncompressed.zarrmd",
    n_atoms=u.trajectory.n_atoms,
    n_frames=u.trajectory.n_frames,
    precision=3,
    compressor=numcodecs.Blosc(cname="zstd", clevel=0),
) as W:
    for ts in u.trajectory:
        W.write(u.atoms)
