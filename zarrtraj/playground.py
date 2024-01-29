
from ZARRTRAJ import *
import MDAnalysis as mda
from MDAnalysisTests.datafiles import PSF
import zarr
import numpy as np
import os
import h5py

import fsspec
import s3fs
import zarr





u = mda.Universe(PSF, 's3://test-zarrtraj-bucket/zarr_3341_100.zarrtraj', storage_options={'key':'AKIAUODTGZQXMD5QNMP5', 'secret':'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'})
for ts in u.trajectory:
    print(ts[0])