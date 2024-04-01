"""
Unit and regression test for the zarrtraj package when the zarr group is on-disk.
"""

import zarrtraj
from zarrtraj import HAS_ZARR
if HAS_ZARR:
    import zarr
from MDAnalysisTests.dummy import make_Universe
import pytest
from zarrtraj.tests.datafiles import COORDINATES_ZARRTRAJ#, ZARRTRAJ_xvf
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
from MDAnalysisTests.datafiles import (TPR_xvf, TRR_xvf,
                                       COORDINATES_TOPOLOGY)
from MDAnalysisTests.coordinates.base import (MultiframeReaderTest,
                                              BaseReference, BaseWriterTest,
                                              assert_timestep_almost_equal)
import MDAnalysis as mda
import sys
from .conftest import ZARRTRAJReference

@pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")
class TestZarrTrajReaderBaseAPI(MultiframeReaderTest):
    """Tests ZarrTrajReader with with synthetic trajectory."""
    @staticmethod
    @pytest.fixture()
    def ref():
        return ZARRTRAJReference()

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