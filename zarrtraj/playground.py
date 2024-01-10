from Zarr import ZarrReader
import MDAnalysis as mda
import zarr
import numpy as np


# Create an initialized array (change np.zeros to np.ones if you want ones)
data = np.zeros([3, 3])

# Open the Zarr file for writing
a = zarr.open_array(path='/home/law/workspace/zarrtraj/all_ones.zarr', mode='w-', shape=data.shape, dtype=data.dtype)

# Assign data to the Zarr array
a[:] = data

# Print details
print(a)
print(a.shape)

# Load the array to check
loaded = zarr.load('/home/law/workspace/zarrtraj/all_ones.zarr')
print(loaded)