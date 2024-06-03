import fsspec
import kerchunk.hdf
import ujson
import xarray as xr
import os

os.environ["S3_REGION_NAME"] = "us-west-1"
os.environ["AWS_PROFILE"] = "sample_profile"

so = dict(anon=False, default_fill_cache=False, default_cache_type="first")
url = "s3://zarrtraj-test-data/peg.h5md"
# storage_options = dict(default_fill_cache=False, default_cache_type="none")
#
#
# with fsspec.open(url, **storage_options) as inf:
#    ref = kerchunk.hdf.SingleHdf5ToZarr(
#        inf, url, inline_threshold=0
#    ).translate()
#    # now that we have this .zmetadata key, we
#    # need to actually write it to the h5md file
#    with open("reference.json", "w") as f:
#        json.dump(ref, f)
#
#

with fsspec.open(url, **so) as inf:
    h5chunks = kerchunk.hdf.SingleHdf5ToZarr(inf, url, inline_threshold=100)
    h5chunks.translate()
    with open("single_file_kerchunk.json", "wb") as f:
        f.write(ujson.dumps(h5chunks.translate()).encode())


fs = fsspec.filesystem(
    "reference",
    fo="single_file_kerchunk.json",
    remote_protocol="s3",
    remote_options=dict(anon=False),
    skip_instance_cache=True,
)
# m = fsspec.get_mapper("reference://", fo="reference.json", remote_protocol="s3")
ds = xr.open_datatree(
    fs.get_mapper(""), engine="zarr", backend_kwargs={"consolidated": False}
)
print(ds)
