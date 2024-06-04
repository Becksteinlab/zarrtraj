import fsspec
import kerchunk.hdf
import ujson
import xarray as xr
import os
import datatree

os.environ["S3_REGION_NAME"] = "us-west-1"
os.environ["AWS_PROFILE"] = "sample_profile"

so = dict(anon=False, default_fill_cache=False, default_cache_type="first")
url = "s3://zarrtraj-test-data/long_0_all_100.zarrtraj"

# m = fsspec.get_mapper("reference://", fo="reference.json", remote_protocol="s3")
ds = datatree.open_datatree(
    url, engine="zarr", backend_kwargs={"consolidated": False})
print(ds)
print(ds["particles"])