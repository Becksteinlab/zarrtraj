import boto3
import zarr


file = zarr.open_group(
    "s3://zarrtraj-test-data/COORDINATES_SYNTHETIC_ZARRMD.zarrmd", mode="r"
)
print(zarr.tree())
