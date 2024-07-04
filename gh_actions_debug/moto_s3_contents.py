import boto3
import zarr
import os

os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SECURITY_TOKEN"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"

# For convenience, set dict options as env vars
# boto options
os.environ["AWS_DEFAULT_REGION"] = "us-west-1"
os.environ["AWS_ENDPOINT_URL"] = "http://localhost:5000"
# s3fs options
os.environ["S3_REGION_NAME"] = "us-west-1"
os.environ["S3_ENDPOINT_URL"] = "http://localhost:5000"
filename = "s3://zarrtraj-test-data/COORDINATES_SYNTHETIC_ZARRMD.zarrmd"

mapping = zarr.storage.FSStore(filename, mode="r")
cache = zarr.storage.LRUStoreCache(mapping, max_size=(100 * 1024**2))
file = zarr.open_group(store=cache, mode="r")

print(file.tree())
print(len(list(file["particles"].group_keys())))
