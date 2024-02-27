# TEST- New buffered writer

from ZARRTRAJ import *
from MDAnalysisTests.datafiles import PSF, DCD
import fsspec
import s3fs
import os
import time
import MDAnalysis as mda
import zarr

import MDAnalysisData

yiip = MDAnalysisData.yiip_equilibrium.fetch_yiip_equilibrium_short()
#key = os.getenv('AWS_KEY')
#secret = os.getenv('AWS_SECRET_KEY')
key = "AKIA6RJXOAIBRK4FNSWI"
secret = "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"
s3 = s3fs.S3FileSystem(key=key, secret=secret)
store = s3fs.S3Map(root='zarrtraj-test-data/s3-yiip-test.zarrtraj', s3=s3, check=False)
z = zarr.open_group(store=store, mode='w')

u = mda.Universe(yiip.topology, yiip.trajectory)
start = time.time()
with mda.Writer(z, u.trajectory.n_atoms, n_frames=u.trajectory.n_frames, format='ZARRTRAJ', chunks=(100, u.trajectory.n_atoms, 3)) as w:
    for ts in u.trajectory:
        w.write(u.atoms)
stop = time.time()
print(f"Total writing time is {stop-start} seconds")
