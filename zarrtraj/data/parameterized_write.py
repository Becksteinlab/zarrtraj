"""Used to generate benchmarking data for asv"""
import zarrtraj
import zarr
from numcodecs import Blosc, Quantize
import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD

uShort = mda.Universe(PSF, DCD)

compressor_lvl = [0, 1, 9]
filters_list = ["all", 3]
chunks_frames = [1, 10, 50]
# total traj is 1.3 MB
# 98 frames, 3341 atoms

for c in compressor_lvl:
    for f in filters_list:
        for ch in chunks_frames:
            if f == 3:
                filters = [Quantize(digits=3, dtype='f4')]
            else:
                filters = None
            z = zarr.open_group(f"short_{c}_{f}_{ch}.zarrtraj")
            with mda.Writer(z, n_atoms=uShort.trajectory.n_atoms, 
                            n_frames=uShort.trajectory.n_frames,
                            force_buffered=True,
                            compressor=Blosc(cname='zstd', clevel=c),
                            filters=filters,
                            chunks=(ch, uShort.trajectory.n_atoms, 3),
                            format="ZARRTRAJ"
                            ) as w:
                for ts in uShort.trajectory:
                    w.write(uShort)