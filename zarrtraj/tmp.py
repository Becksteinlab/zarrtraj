import fsspec
import kerchunk.hdf
import ujson
import xarray as xr
import os
import zarr

os.environ["S3_REGION_NAME"] = "us-west-1"
os.environ["AWS_PROFILE"] = "sample_profile"

so = dict(anon=False, default_fill_cache=False, default_cache_type="first")
url = "s3://zarrtraj-test-data/peg.h5md"

with fsspec.open(url, **so) as inf:
    h5chunks = kerchunk.hdf.SingleHdf5ToZarr(inf, url, inline_threshold=100)
    with open("single_file_kerchunk.json", "wb") as f:
        f.write(ujson.dumps(h5chunks.translate()).encode())


fs = fsspec.filesystem(
    "reference",
    fo="single_file_kerchunk.json",
    remote_protocol="s3",
    remote_options=dict(anon=False),
    skip_instance_cache=True,
)

z = zarr.open_group(fs.get_mapper(""), mode="r")

print(z.tree())
