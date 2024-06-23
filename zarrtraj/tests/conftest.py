"""
Global pytest fixtures
"""

# Use this file if you need to share any fixtures
# across multiple modules
# More information at
# https://docs.pytest.org/en/stable/how-to/fixtures.html#scope-sharing-fixtures-across-classes-modules-packages-or-session

import pytest
import zarrtraj

from moto.server import ThreadedMotoServer
import os
import boto3


@pytest.fixture(scope="session", autouse=True)
def moto_server():
    """Start moto server"""
    print("Starting moto server")
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

    # Using boto3.resource rather than .client since we don't
    # Need granular control
    s3_resource = boto3.resource("s3")
    s3_resource.create_bucket(
        Bucket="zarrtraj-test-data",
        CreateBucketConfiguration={"LocationConstraint": "us-west-1"},
    )
    yield
    server.stop()
