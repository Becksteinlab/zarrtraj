import zarr
import zarrtraj
from zarrtraj.tests.datafiles import *
import MDAnalysis as mda
from MDAnalysisTests.datafiles import (TPR_xvf, TRR_xvf,
                                       COORDINATES_TOPOLOGY)


print(zarrtraj.__path__)

#
# 
z = zarr.open_group(COORDINATES_ZARRTRAJ)

print(z.tree())
#print(COORDINATES_ZARRTRAJ)

#u = mda.Universe(COORDINATES_TOPOLOGY, zarr.open_group(COORDINATES_ZARRTRAJ, 'r'))
#
#for ts in u:
#    print(ts)
