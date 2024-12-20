import zarrtraj
import zarr
from numcodecs import Blosc, Quantize
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
import numcodecs
import os

os.environ["AWS_PROFILE"] = "sample_profile"
os.environ["AWS_REGION"] = "us-west-1"

u = mda.Universe(
    "zarrtraj/data/yiip_equilibrium/YiiP_system.pdb",
    "zarrtraj/data/yiip_equilibrium/YiiP_system_90ns_center_aligned.xtc",
)

with mda.Writer(
    "s3://zarrtraj-test-data/yiip_aligned_uncompressed.zarrmd",
    n_atoms=u.trajectory.n_atoms,
    n_frames=u.trajectory.n_frames,
    precision=3,
    compressor=numcodecs.Blosc(cname="zstd", clevel=0),
) as W:
    for ts in u.trajectory:
        W.write(u.atoms)
