"""
Unit and regression test for the zarrtraj package when the zarr group is 
in an AWS S3 bucket
"""

import pytest
import os

# While debugging and server is running, visit http://localhost:5000/moto-api
from moto.server import ThreadedMotoServer
import boto3
import s3fs

import zarrtraj
from zarrtraj import HAS_ZARR

if HAS_ZARR:
    import zarr
from MDAnalysisTests.dummy import make_Universe
from zarrtraj.tests.datafiles import COORDINATES_ZARRTRAJ, ZARRTRAJ_xvf
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
import MDAnalysis as mda
from MDAnalysisTests.datafiles import TPR_xvf, TRR_xvf, COORDINATES_TOPOLOGY
from MDAnalysisTests.coordinates.base import (
    MultiframeReaderTest,
    BaseReference,
    BaseWriterTest,
    assert_timestep_almost_equal,
    assert_array_almost_equal,
)
from .conftest import ZARRTRAJReference
import requests
import numpy as np


def create_bucket(bucket_name):
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

    # Using boto3.resource rather than .client since we don't
    # Need granular control
    s3_resource = boto3.resource("s3")
    s3_resource.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": "us-west-1"},
    )


# Only call this once a Moto Server is running
def put_zarrtraj_in_bucket(fname, bucket_name):
    source = zarr.open_group(fname, mode="r")

    s3_fs = s3fs.S3FileSystem(
        anon=False,
    )
    cloud_store = s3fs.S3Map(
        root=f"{bucket_name}/{os.path.basename(fname)}", s3=s3_fs, check=False
    )

    zarr.convenience.copy_store(source.store, cloud_store, if_exists="replace")

    return "s3://" + bucket_name + "/" + os.path.basename(fname)


@pytest.mark.skipif(not HAS_ZARR, reason="Zarr not installed")
class ZARRTRAJAWSReference(BaseReference):
    """Reference synthetic trajectory that was
    copied from test_xdr.TRRReference"""

    def __init__(self):
        super(ZARRTRAJAWSReference, self).__init__()
        self.trajectory = put_zarrtraj_in_bucket(
            COORDINATES_ZARRTRAJ, "test-read-bucket"
        )
        self.topology = COORDINATES_TOPOLOGY
        self.reader = zarrtraj.ZarrTrajReader
        self.writer = zarrtraj.ZarrTrajWriter
        self.ext = "zarrtraj"
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


@pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")
class TestZarrTrajAWSReaderBaseAPI(MultiframeReaderTest):
    """Tests ZarrTrajReader with with synthetic trajectory."""

    @pytest.fixture(scope="class")
    def bucketname(self):
        return "test-read-bucket"

    @pytest.fixture(autouse=True, scope="class")
    def run_server(self, bucketname):
        self.server = ThreadedMotoServer()
        self.server.start()
        create_bucket(bucketname)
        yield
        self.server.stop()

    # Only create one ref to avoid high memory usage
    @pytest.fixture(scope="class")
    def ref(self):
        r = ZARRTRAJAWSReference()
        yield r

    @staticmethod
    @pytest.fixture()
    def reader(ref):
        reader = ref.reader(ref.trajectory)
        reader.add_auxiliary(
            "lowf",
            ref.aux_lowf,
            dt=ref.aux_lowf_dt,
            initial_time=0,
            time_selector=None,
        )
        reader.add_auxiliary(
            "highf",
            ref.aux_highf,
            dt=ref.aux_highf_dt,
            initial_time=0,
            time_selector=None,
        )
        yield reader

    def test_get_writer_1(self, ref, reader, tmpdir, bucketname):
        with tmpdir.as_cwd():
            outfn = "s3://" + bucketname + "/test-writer.zarrtraj"
            with reader.Writer(outfn) as W:
                assert_equal(isinstance(W, ref.writer), True)
                assert_equal(W.n_atoms, reader.n_atoms)

    def test_get_writer_2(self, ref, reader, tmpdir, bucketname):
        with tmpdir.as_cwd():
            outfn = "s3://" + bucketname + "/test-writer.zarrtraj"
            with reader.Writer(outfn, n_atoms=100) as W:
                assert_equal(isinstance(W, ref.writer), True)
                assert_equal(W.n_atoms, 100)

    def test_copying(self, ref):
        original = zarrtraj.ZarrTrajReader(
            ref.trajectory, dt=2, time_offset=10, foo="bar"
        )
        copy = original.copy()

        assert original.format not in ("MEMORY", "CHAIN")
        assert original._ts_kwargs["time_offset"] == 10
        assert copy._ts_kwargs["time_offset"] == 10
        assert original._ts_kwargs["dt"] == 2
        assert copy._ts_kwargs["dt"] == 2

        assert original.ts.data["time_offset"] == 10
        assert copy.ts.data["time_offset"] == 10

        assert original.ts.data["dt"] == 2
        assert copy.ts.data["dt"] == 2

        assert copy._kwargs["foo"] == "bar"

        # check coordinates
        assert original.ts.frame == copy.ts.frame
        assert_allclose(original.ts.positions, copy.ts.positions)

        original.next()
        copy.next()

        assert original.ts.frame == copy.ts.frame
        assert_allclose(original.ts.positions, copy.ts.positions)


@pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")
class TestZarrTrajAWSWriterBaseAPI(BaseWriterTest):
    """Tests ZarrTrajWriter with with synthetic trajectory."""

    @pytest.fixture(scope="class")
    def bucketname(self):
        return "test-write-bucket"

    # Run one moto server for the entire class
    # And only keep one zarr group to write to
    @pytest.fixture(autouse=True, scope="class")
    def run_server(self, bucketname):
        self.server = ThreadedMotoServer()
        self.server.start()
        create_bucket(bucketname)
        yield
        self.server.stop()

    @pytest.fixture()
    def outfn(self, bucketname):
        r = "s3://" + bucketname + "/test-write.zarrtraj"
        yield r
        zarr.storage.rmdir(r)

    @staticmethod
    @pytest.fixture()
    def ref():
        yield ZARRTRAJReference()

    def test_write_different_box(self, ref, universe, outfn):
        if ref.changing_dimensions:
            with ref.writer(
                outfn,
                universe.atoms.n_atoms,
                n_frames=universe.trajectory.n_frames,
                # must use force_buffered here because
                # on GH actions runners, store type
                # resolves to KVStore rather than FSStore
                force_buffered=True,
            ) as W:
                for ts in universe.trajectory:
                    universe.dimensions[:3] += 1
                    W.write(universe)
            written = ref.reader(outfn)

            for ts_ref, ts_w in zip(universe.trajectory, written):
                universe.dimensions[:3] += 1
                assert_array_almost_equal(
                    universe.dimensions, ts_w.dimensions, decimal=ref.prec
                )

    def test_write_selection(
        self,
        ref,
        reader,
        universe,
        u_no_resnames,
        u_no_resids,
        u_no_names,
        outfn,
    ):
        sel_str = "resid 1"
        sel = universe.select_atoms(sel_str)

        with ref.writer(
            outfn,
            sel.n_atoms,
            n_frames=universe.trajectory.n_frames,
            force_buffered=True,
        ) as W:
            for ts in universe.trajectory:
                W.write(sel.atoms)
        copy = ref.reader(outfn)
        for orig_ts, copy_ts in zip(universe.trajectory, copy):
            assert_array_almost_equal(
                copy_ts._pos,
                sel.atoms.positions,
                ref.prec,
                err_msg="coordinate mismatch between original and written "
                "trajectory at frame {} (orig) vs {} (copy)".format(
                    orig_ts.frame, copy_ts.frame
                ),
            )

    def test_write_none(self, ref, outfn):
        with pytest.raises(TypeError):
            with ref.writer(
                outfn,
                42,
                n_frames=1,
                max_memory=1,
                chunks=(1, 5, 3),
            ) as w:
                w.write(None)

    def test_write_not_changing_ts(self, ref, universe, outfn):
        copy_ts = universe.trajectory.ts.copy()
        with ref.writer(
            outfn,
            n_atoms=5,
            n_frames=1,
            force_buffered=True,
        ) as W:
            W.write(universe)
            assert_timestep_almost_equal(copy_ts, universe.trajectory.ts)

    def test_write_trajectory_atomgroup(self, ref, reader, universe, outfn):
        with ref.writer(
            outfn,
            universe.atoms.n_atoms,
            n_frames=universe.trajectory.n_frames,
            force_buffered=True,
        ) as w:
            for ts in universe.trajectory:
                w.write(universe.atoms)
        self._check_copy(outfn, ref, reader)

    def test_write_trajectory_universe(self, ref, reader, universe, outfn):
        with ref.writer(
            outfn,
            universe.atoms.n_atoms,
            n_frames=universe.trajectory.n_frames,
            force_buffered=True,
        ) as w:
            for ts in universe.trajectory:
                w.write(universe)
        self._check_copy(outfn, ref, reader)
