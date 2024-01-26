
import MDAnalysis as mda
import zarr
import numpy as np
import os
import h5py

print(os.getcwd())

z = zarr.open_group('zarrtraj/10e3_zarr_c.zarrtraj', 'r')
print(z)