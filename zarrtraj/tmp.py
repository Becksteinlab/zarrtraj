import fsspec
import kerchunk.hdf
import ujson
import os
import zarr

os.environ["S3_REGION_NAME"] = "us-west-1"
os.environ["AWS_PROFILE"] = "sample_profile"

so = dict(anon=False, default_fill_cache=False, default_cache_type="first")
url = "s3://zarrtraj-test-data/peg.h5md"

with fsspec.open(url, **so) as inf:
    h5chunks = kerchunk.hdf.SingleHdf5ToZarr(inf, url, inline_threshold=100)
    fo = h5chunks.translate()


fs = fsspec.filesystem(
    "reference",
    fo=fo,
    remote_protocol="s3",
    remote_options=so,
    skip_instance_cache=True,
)

z = zarr.open_group(fs.get_mapper(""), mode="r")
print(z.particles)
