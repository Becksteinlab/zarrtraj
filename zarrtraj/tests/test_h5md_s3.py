import zarrtraj
import zarr
import os
import boto3
import s3fs

import pytest
from botocore.exceptions import ClientError
from MDAnalysis.exceptions import NoDataError
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
    COORDINATES_MISSING_H5MD_GROUP_H5MD,
    COORDINATES_MISSING_H5MD_GROUP_ZARRMD,
    COORDINATES_MISSING_TIME_DSET_H5MD,
    COORDINATES_MISSING_TIME_DSET_ZARRMD,
    COORDINATES_VARIED_STEPS_H5MD,
    COORDINATES_VARIED_STEPS_ZARRMD,
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
@pytest.mark.parametrize(
    "file",
    [
        COORDINATES_MISSING_H5MD_GROUP_H5MD,
        COORDINATES_MISSING_H5MD_GROUP_ZARRMD,
    ],
)
def test_missing_h5md_grp(file):
    with pytest.raises(
        ValueError, match="H5MD file must contain an 'h5md' group"
    ):
        zarrtraj.ZARRH5MDReader(file)


@pytest.mark.parametrize(
    "file",
    [
        COORDINATES_MISSING_TIME_DSET_H5MD,
        COORDINATES_MISSING_TIME_DSET_ZARRMD,
    ],
)
def test_missing_time_dset(file):
    """Time datasets are optional in the H5MD format, however,
    MDAnalysis requires that time data is available for each sampled
    integration step"""
    with pytest.raises(
        NoDataError, match="MDAnalysis requires that time data is available "
    ):
        zarrtraj.ZARRH5MDReader(file)


@pytest.mark.parametrize(
    "file",
    [
        COORDINATES_VARIED_STEPS_H5MD,
        COORDINATES_VARIED_STEPS_ZARRMD,
    ],
)
class TestH5MDFmtReaderVariedSteps(object):
    """Try to break the reader with a ridiculous but technically
    up-to-standard H5MD file."""

    @pytest.fixture
    def true_vals(self):
        yield zarrtraj.ZARRH5MDReader(COORDINATES_SYNTHETIC_H5MD)

    @pytest.fixture
    def reader(self, file):
        return zarrtraj.ZARRH5MDReader(file)

    def test_global_step_time(self, reader):
        # Ensure global step array was constructed correctly
        # xvf sampled up to integration step 4 but
        # observables sampled at integration step 5,
        # so step and time should have length 5
        for i in range(6):
            assert reader[i].time == i
            assert reader[i].data["step"] == i

    def test_explicit_offset_velocities(self, reader, true_vals):
        for i in range(3):
            assert_array_almost_equal(
                reader[i].velocities, true_vals[i].velocities
            )
        for i in range(3, 6):
            assert not reader[i].has_velocities

    def test_fixed_offset_forces(self, reader, true_vals):
        assert not reader[0].has_forces
        for i in range(1, 5):
            assert_array_almost_equal(reader[i].forces, true_vals[i].forces)
        assert not reader[5].has_forces

    def test_fixed_positions(self, reader, true_vals):
        for i in range(5):
            assert_array_almost_equal(
                reader[i].positions, true_vals[i].positions
            )
        assert not reader[5].has_positions

    def test_particle_group_observables(self, reader):
        for i in range(1, 6, 2):
            assert reader[i].data["trajectory/obsv1"] == i + 3
            assert reader[i].data["trajectory/obsv2"] == i + 3
        assert "trajectory/obsv1" not in reader[0].data
        assert "trajectory/obsv2" not in reader[0].data
        for i in range(0, 6, 2):
            assert "trajectory/obsv1" not in reader[i].data
            assert "trajectory/obsv2" not in reader[i].data

    def test_global_observables(self, reader):
        for i in range(1, 6, 2):
            assert reader[i].data["obsv1"] == i + 3
            assert reader[i].data["obsv2"] == i + 3
        assert "obsv1" not in reader[0].data
        assert "obsv2" not in reader[0].data
        for i in range(0, 6, 2):
            assert "obsv1" not in reader[i].data
            assert "obsv2" not in reader[i].data

    def test_time_independent_observables(self, reader):
        for i in range(6):
            assert_array_almost_equal(reader[i].data["obsv3"], [4, 6, 8])
            assert_array_almost_equal(
                reader[i].data["trajectory/obsv3"], [4, 6, 8]
            )
