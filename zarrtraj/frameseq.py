import zarrtraj
import MDAnalysis as mda
from zarrtraj.tests.datafiles import COORDINATES_ZARRTRAJ
from MDAnalysisTests.datafiles import COORDINATES_TOPOLOGY
import zarr

# u = mda.Universe("PEG_1chain/PEG.prmtop", "peg.hdf5")

z = zarr.open_group(COORDINATES_ZARRTRAJ)

n = z["particles/positions"].name

print(z[n])
