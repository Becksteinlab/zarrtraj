import fsspec
import kerchunk.hdf
import ujson
import os
import zarr


url = "peg.hdf5"

with fsspec.open(url) as inf:
    h5chunks = kerchunk.hdf.SingleHdf5ToZarr(inf, url, inline_threshold=100)
    fo = h5chunks.translate()


fs = fsspec.filesystem(
    "reference",
    fo=fo,
    skip_instance_cache=True,
)

z = zarr.open_group(fs.get_mapper(""), mode="r")
print(z.particles.trajectory)
