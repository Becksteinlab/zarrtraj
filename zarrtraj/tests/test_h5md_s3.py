import zarrtraj
import zarr
import os
import boto3
import s3fs

import pytest
from botocore.exceptions import ClientError
import logging
from MDAnalysisTests.coordinates.base import (
    MultiframeReaderTest,
    BaseReference,
    BaseWriterTest,
    assert_timestep_almost_equal,
    assert_array_almost_equal,
)
from MDAnalysisTests.datafiles import (
    H5MD_xvf,
    TPR_xvf,
    TRR_xvf,
    COORDINATES_TOPOLOGY,
)
from zarrtraj.tests.datafiles import (
    COORDINATES_SYNTHETIC_H5MD,
    COORDINATES_SYNTHETIC_ZARRMD,
)


def upload_h5md_testfile(file_name):
    s3_client = boto3.client("s3")
    obj_name = os.path.basename(file_name)
    try:
        response = s3_client.upload_file(
            file_name, "zarrtraj-test-data", obj_name
        )
    except ClientError as e:
        logging.error(e)
        return False
    return True


def upload_zarrmd_testfile(file_name):
    source = zarr.open_group(file_name, mode="r")
    obj_name = os.path.basename(file_name)
    s3_fs = s3fs.S3FileSystem()
    cloud_store = s3fs.S3Map(
        root=f"s3://zarrtraj-test-data/{obj_name}", s3=s3_fs
    )

    zarr.convenience.copy_store(source.store, cloud_store, if_exists="replace")

    return True


class H5MDFmtReference(BaseReference):
    """Reference synthetic trajectory that was
    copied from test_xdr.TRRReference"""

    def __init__(self, filename):
        super(H5MDFmtReference, self).__init__()
        self.trajectory = filename
        self.topology = COORDINATES_TOPOLOGY
        self.reader = zarrtraj.ZARRH5MDReader
        # self.writer = zarrtraj.ZarrTrajWriter
        self.ext = "zarrmd"
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


@pytest.fixture(scope="class")
def ref(request):
    store, filename = request.param
    if store == "s3":
        if filename.endswith(".h5md"):
            upload_h5md_testfile(filename)
        elif filename.endswith(".zarrmd"):
            upload_zarrmd_testfile(filename)
        s3fn = "s3://zarrtraj-test-data/" + os.path.basename(filename)
        yield H5MDFmtReference(s3fn)
    else:
        yield H5MDFmtReference(filename)


@pytest.mark.parametrize(
    "ref",
    [
        ("s3", COORDINATES_SYNTHETIC_H5MD),
        ("local", COORDINATES_SYNTHETIC_H5MD),
        ("s3", COORDINATES_SYNTHETIC_ZARRMD),
        ("local", COORDINATES_SYNTHETIC_ZARRMD),
    ],
    indirect=True,
)
class TestH5MDFmtReaderBaseAPI(MultiframeReaderTest):
    """Tests ZarrTrajReader with with synthetic trajectory."""


# H5MD Format Reader Tests
# Only test these with disk to avoid excessive test length
# @pytest.mark.parametrize(
#    "file",
#    [COORDINATES_MISSING_H5MD_GRP_H5MD, COORDINATES_MISSING_H5MD_GRP_ZARRMD],
# )
# class TestH5MDFmtReaderMissingH5MDGrp:
#    def test_missing_h5md_grp(self, file):
#        with pytest.raises(ValueError):
#            zarrtraj.ZARRH5MDReader(file)
