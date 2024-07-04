from moto.server import ThreadedMotoServer
import os
from zarrtraj.tests.datafiles import COORDINATES_SYNTHETIC_ZARRMD
import zarr
import s3fs

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

server = ThreadedMotoServer()
server.start()

# upload file
source = zarr.open_group(COORDINATES_SYNTHETIC_ZARRMD, mode="r")
obj_name = os.path.basename(COORDINATES_SYNTHETIC_ZARRMD)
s3_fs = s3fs.S3FileSystem()
cloud_store = s3fs.S3Map(root=f"s3://zarrtraj-test-data/{obj_name}", s3=s3_fs)

zarr.convenience.copy_store(source.store, cloud_store, if_exists="raise")
