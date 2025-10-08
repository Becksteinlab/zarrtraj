import os
import boto3

os.environ["AWS_PROFILE"] = "sample_profile"
os.environ["AWS_REGION"] = "us-west-1"

def upload_h5md_file(bucket_name, file_name):
    s3_client = boto3.client("s3")
    obj_name = os.path.basename(file_name)

    response = s3_client.upload_file(file_name, bucket_name, obj_name)

upload_h5md_file("zarrtraj-test-data", "zarrtraj/data/yiip_aligned_uncompressed.h5md")
