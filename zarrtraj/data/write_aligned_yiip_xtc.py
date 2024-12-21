import zarrtraj
import zarr
from numcodecs import Blosc, Quantize
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
import numcodecs

# This requires MDAnalysis >= 2.8.0

u = mda.Universe(
    "zarrtraj/data/yiip_equilibrium/YiiP_system.pdb",
    "zarrtraj/data/yiip_equilibrium/YiiP_system_90ns_center.xtc",
)

average = align.AverageStructure(
    u, u, select="protein and name CA", ref_frame=0
).run()
ref = average.results.universe

aligner = align.AlignTraj(
    u,
    ref,
    select="protein and name CA",
    filename="zarrtraj/data/yiip_equilibrium/YiiP_system_90ns_center_aligned.xtc",
).run()
