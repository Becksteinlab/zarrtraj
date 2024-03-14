"""
Unit and regression test for the zarrtraj package when the zarr group is 
in an AWS S3 bucket
"""

import pytest
import os
from moto import mock_s3
import boto3
import s3fs

import zarrtraj
from zarrtraj import HAS_ZARR
if HAS_ZARR:
    import zarr
from MDAnalysisTests.dummy import make_Universe
from zarrtraj.tests.datafiles import COORDINATES_ZARRTRAJ#, ZARRTRAJ_xvf
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
from MDAnalysisTests.datafiles import (TPR_xvf, TRR_xvf,
                                       COORDINATES_TOPOLOGY)
from MDAnalysisTests.coordinates.base import (MultiframeReaderTest,
                                              BaseReference, BaseWriterTest,
                                              assert_timestep_almost_equal)
import sys


@pytest.fixture
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    yield


@pytest.fixture(scope="function")
@mock_s3
def zarr_file_to_s3_bucket(aws_credentials):
    def _setup_bucket_with_zarr_group(fname):
        # Using boto3.resource rather than .client since we don't
        # Need granular control
        s3_resource = boto3.resource("s3", region_name="us-east-1")
        s3_resource.create_bucket(Bucket="testbucket")

        source = zarr.open_group(fname, mode='r')

        s3_fs = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name='us-east-1'))
        store = s3fs.S3Map(root=f'testbucket/{COORDINATES_ZARRTRAJ}',
                           s3=s3_fs, check=False)
        cloud_dest = zarr.open_group(store=store, mode='r')

        zarr.convenience.copy_store(source, cloud_dest)
        return cloud_dest

@pytest.mark.skipif(not HAS_ZARR, reason="Zarr not installed")
@mock_s3
class ZARRTRAJReference(BaseReference):
    """Reference synthetic trajectory that was
    copied from test_xdr.TRRReference"""

    def __init__(self):
        super(ZARRTRAJReference, self).__init__()
        self.trajectory = zarr_file_to_s3_bucket(COORDINATES_ZARRTRAJ)
        self.topology = COORDINATES_TOPOLOGY
        self.reader = zarrtraj.ZarrTrajReader
        self.writer = zarrtraj.ZarrTrajWriter
        self.ext = 'zarrtraj'
        self.prec = 3
        self.changing_dimensions = True

        self.first_frame.velocities = self.first_frame.positions / 10
        self.first_frame.forces = self.first_frame.positions / 100

        self.second_frame.velocities = self.second_frame.positions / 10
        self.second_frame.forces = self.second_frame.positions / 100

        self.last_frame.velocities = self.last_frame.positions / 10
        self.last_frame.forces = self.last_frame.positions / 100

        self.jump_to_frame.velocities = self.jump_to_frame.positions / 10
        self.jump_to_frame.forces = self.jump_to_frame.positions / 100

    def iter_ts(self, i):
        ts = self.first_frame.copy()
        ts.positions = 2**i * self.first_frame.positions
        ts.velocities = ts.positions / 10
        ts.forces = ts.positions / 100
        ts.time = i
        ts.frame = i
        return ts