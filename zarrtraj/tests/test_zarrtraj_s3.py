"""
Unit and regression test for the zarrtraj package when the zarr group is 
in an AWS S3 bucket
"""

import pytest
import os
from moto import mock_aws
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
from MDAnalysisTests.datafiles import (TPR_xvf, TRR_xvf,
                                       COORDINATES_TOPOLOGY)
from MDAnalysisTests.coordinates.base import (MultiframeReaderTest,
                                              BaseReference, BaseWriterTest,
                                              assert_timestep_almost_equal,
                                              assert_array_almost_equal)
from .conftest import ZARRTRAJReference
import requests
# Must ensure unique bucket name is created for GH actions
import uuid

# Only call this once a Moto Server is running
def zarr_file_to_s3_bucket(fname):
    bucket_name = f"testbucket-{uuid.uuid4()}"

    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"

    # Using boto3.resource rather than .client since we don't
    # Need granular control
    s3_resource = boto3.resource(
        "s3",
        region_name="us-west-1",
        endpoint_url="http://localhost:5000"
    )
    s3_resource.create_bucket(Bucket=bucket_name)

    source = zarr.open_group(fname, mode='r')

    s3_fs = s3fs.S3FileSystem(
        anon=False,
        client_kwargs=dict(
            region_name='us-west-1',
            endpoint_url="http://localhost:5000"
        )
    )
    cloud_store = s3fs.S3Map(
        root=f'{bucket_name}/{os.path.basename(fname)}',
        s3=s3_fs,
        check=False
    )

    zarr.convenience.copy_store(source.store, cloud_store,
                                if_exists='replace')

    cloud_dest = zarr.open_group(store=cloud_store, mode='a')
    return cloud_dest

# Only call this once a Moto Server is running
def new_zarrgroup_in_bucket(fname):
    bucket_name = f"testbucket-{uuid.uuid4()}"

    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"

    # Using boto3.resource rather than .client since we don't
    # Need granular control
    s3_resource = boto3.resource(
        "s3",
        region_name="us-west-1",
        endpoint_url="http://localhost:5000"
    )
    s3_resource.create_bucket(Bucket=bucket_name)

    s3_fs = s3fs.S3FileSystem(
        anon=False,
        client_kwargs=dict(
            region_name='us-west-1',
            endpoint_url="http://localhost:5000"
        )
    )
    cloud_store = s3fs.S3Map(
        root=f'{bucket_name}/{os.path.basename(fname)}',
        s3=s3_fs,
        check=False
    )

    cloud_dest = zarr.open_group(store=cloud_store, mode='a')

    return cloud_dest

# Helper function to calculate the memory usage of a writer at a frame
def get_memory_usage(writer):
    mem = (writer._time_buffer.nbytes + writer._step_buffer.nbytes + 
           writer._edges_buffer.nbytes + writer._pos_buffer.nbytes + 
           writer._force_buffer.nbytes + writer._vel_buffer.nbytes)
    return mem

@pytest.mark.skipif(not HAS_ZARR, reason="Zarr not installed")
class ZARRTRAJAWSReference(BaseReference):
    """Reference synthetic trajectory that was
    copied from test_xdr.TRRReference"""
    def __init__(self):
        super(ZARRTRAJAWSReference, self).__init__()
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


@pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")
class TestZarrTrajAWSReaderBaseAPI(MultiframeReaderTest):
    """Tests ZarrTrajReader with with synthetic trajectory."""

    @pytest.fixture(autouse=True, scope='class')
    def run_server(self):
        self.server = ThreadedMotoServer()
        self.server.start()
        yield
        self.server.stop()

    @pytest.fixture(autouse=True)
    def reset_server(self):
        yield
        requests.post("http://localhost:5000/moto-api/reset")

    @staticmethod
    @pytest.fixture()
    def ref():
        yield ZARRTRAJAWSReference()



    def test_get_writer_1(self, ref, reader, tmpdir):
        with tmpdir.as_cwd():
            outfile = zarr.open_group('test-writer' + ref.ext, 'a')
            with reader.Writer(outfile) as W:
                assert_equal(isinstance(W, ref.writer), True)
                assert_equal(W.n_atoms, reader.n_atoms)

    def test_get_writer_2(self, ref, reader, tmpdir):
        with tmpdir.as_cwd():
            outfile = zarr.open_group('test-writer' + ref.ext, 'a')
            with reader.Writer(outfile, n_atoms=100) as W:
                assert_equal(isinstance(W, ref.writer), True)
                assert_equal(W.n_atoms, 100)

    def test_copying(self, ref, reader):
        original = zarrtraj.ZarrTrajReader(
                ref.trajectory, convert_units=False, dt=2,
                time_offset=10, foo="bar")
        copy = original.copy()

        assert original.format not in ('MEMORY', 'CHAIN')
        assert original.convert_units is False
        assert copy.convert_units is False
        assert original._ts_kwargs['time_offset'] == 10
        assert copy._ts_kwargs['time_offset'] == 10
        assert original._ts_kwargs['dt'] == 2
        assert copy._ts_kwargs['dt'] == 2

        assert original.ts.data['time_offset'] == 10
        assert copy.ts.data['time_offset'] == 10

        assert original.ts.data['dt'] == 2
        assert copy.ts.data['dt'] == 2

        assert copy._kwargs['foo'] == 'bar'

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


    @pytest.fixture(autouse=True, scope='class')
    def run_server(self):
        self.server = ThreadedMotoServer()
        self.server.start()
        yield
        self.server.stop()

    @pytest.fixture(autouse=True)
    def reset_server(self):
        yield
        requests.post("http://localhost:5000/moto-api/reset")

    @staticmethod
    @pytest.fixture()
    def ref():
        yield ZARRTRAJReference()

    def test_write_different_box(self, ref, universe, tmpdir):
        if ref.changing_dimensions:
            outfn = 'write-dimensions-test' + ref.ext
            outfile = new_zarrgroup_in_bucket(outfn)
            with tmpdir.as_cwd():
                with ref.writer(outfile, universe.atoms.n_atoms,
                                n_frames=universe.trajectory.n_frames,
                                format='ZARRTRAJ') as W:
                    for ts in universe.trajectory:
                        universe.dimensions[:3] += 1
                        W.write(universe)

                written = ref.reader(outfile)

                for ts_ref, ts_w in zip(universe.trajectory, written):
                    universe.dimensions[:3] += 1
                    assert_array_almost_equal(universe.dimensions,
                                              ts_w.dimensions,
                                              decimal=ref.prec)

    def test_write_selection(self, ref, reader, universe, u_no_resnames,
                             u_no_resids, u_no_names, tmpdir):
        sel_str = 'resid 1'
        sel = universe.select_atoms(sel_str)
        outfn = 'write-selection-test.' + ref.ext
        outfile = new_zarrgroup_in_bucket(outfn)

        with tmpdir.as_cwd():
            with ref.writer(outfile, sel.n_atoms,
                            n_frames=universe.trajectory.n_frames,
                            format='ZARRTRAJ') as W:
                for ts in universe.trajectory:
                    W.write(sel.atoms)

            copy = ref.reader(outfile)
            for orig_ts, copy_ts in zip(universe.trajectory, copy):
                assert_array_almost_equal(
                    copy_ts._pos, sel.atoms.positions, ref.prec,
                    err_msg="coordinate mismatch between original and written "
                            "trajectory at frame {} (orig) vs {} (copy)".format(
                        orig_ts.frame, copy_ts.frame))
                
    def test_write_none(self, ref, tmpdir):
        outfn = 'write-none.' + ref.ext
        outfile = new_zarrgroup_in_bucket(outfn)
        with tmpdir.as_cwd():
            with pytest.raises(TypeError):
                with ref.writer(outfile, 42, n_frames=1,
                                max_memory=1, 
                                chunks=(1, 5, 3), format='ZARRTRAJ') as w:
                    w.write(None)

    def test_write_not_changing_ts(self, ref, universe, tmpdir):
        outfn = 'write-not-changing-ts.' + ref.ext
        outfile = new_zarrgroup_in_bucket(outfn)

        copy_ts = universe.trajectory.ts.copy()
        with tmpdir.as_cwd():
            with ref.writer(outfile, n_atoms=5, n_frames=1,
                            format='ZARRTRAJ') as W:
                W.write(universe)
                assert_timestep_almost_equal(copy_ts, universe.trajectory.ts)

    def test_write_trajectory_atomgroup(self, ref, reader, universe, tmpdir):
        outfn = 'write-atoms-test.' + ref.ext
        outfile = new_zarrgroup_in_bucket(outfn)
        with tmpdir.as_cwd():
            with ref.writer(outfile, universe.atoms.n_atoms,
                            n_frames=universe.trajectory.n_frames,
                            format='ZARRTRAJ') as w:
                for ts in universe.trajectory:
                    w.write(universe.atoms)
            self._check_copy(outfile, ref, reader)

    def test_write_trajectory_universe(self, ref, reader, universe, tmpdir):
        outfn = 'write-uni-test.' + ref.ext
        outfile = new_zarrgroup_in_bucket(outfn)
        with tmpdir.as_cwd():
            with ref.writer(outfile, universe.atoms.n_atoms,
                            n_frames=universe.trajectory.n_frames,
                            format='ZARRTRAJ') as w:
                for ts in universe.trajectory:
                    w.write(universe)
            self._check_copy(outfile, ref, reader)

    def test_max_memory_too_low(self, ref, reader, universe, tmpdir):
        outfn = 'write-max-memory-test.' + ref.ext
        outfile = new_zarrgroup_in_bucket(outfn)
        with tmpdir.as_cwd():
            with pytest.raises(ValueError):
                with ref.writer(outfile, universe.atoms.n_atoms,
                                n_frames=universe.trajectory.n_frames,
                                chunks=(1, universe.trajectory.n_atoms, 3),
                                max_memory=223,
                                format='ZARRTRAJ') as w:
                    for ts in universe.trajectory:
                        w.write(universe)

    def test_max_memory_usage(self, ref, reader, universe, tmpdir):
        outfn = 'write-max-memory-test.' + ref.ext
        outfile = new_zarrgroup_in_bucket(outfn)
        with tmpdir.as_cwd():
            with ref.writer(outfile, universe.atoms.n_atoms,
                            n_frames=universe.trajectory.n_frames,
                            chunks=(1, universe.trajectory.n_atoms, 3),
                            max_memory=224,
                            format='ZARRTRAJ') as w:
                for ts in universe.trajectory:
                    w.write(universe)
                    # Each frame of synthetic trajectory should be 224 bytes
                    assert get_memory_usage(w) <= 224

# Helper Functions
def get_memory_usage(writer):
    mem = (writer._time_buffer.nbytes + writer._step_buffer.nbytes +
           writer._dimensions_buffer.nbytes + writer._pos_buffer.nbytes +
           writer._force_buffer.nbytes + writer._vel_buffer.nbytes)
    return mem