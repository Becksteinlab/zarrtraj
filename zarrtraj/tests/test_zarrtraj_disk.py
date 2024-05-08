"""
Unit and regression test for the zarrtraj package when the zarr group is on-disk.
"""

import zarrtraj
from zarrtraj import HAS_ZARR

if HAS_ZARR:
    import zarr
from MDAnalysisTests.dummy import make_Universe
import pytest
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
import MDAnalysis as mda
import sys
from .conftest import ZARRTRAJReference
from .utils import get_memory_usage, get_frame_size


@pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")
class TestZarrTrajReaderBaseAPI(MultiframeReaderTest):
    """Tests ZarrTrajReader with with synthetic trajectory."""

    @staticmethod
    @pytest.fixture()
    def ref():
        return ZARRTRAJReference()

    def test_get_writer_1(self, ref, reader, tmpdir):
        with tmpdir.as_cwd():
            outfile = zarr.open_group("test-writer" + ref.ext, "a")
            with reader.Writer(outfile) as W:
                assert_equal(isinstance(W, ref.writer), True)
                assert_equal(W.n_atoms, reader.n_atoms)

    def test_get_writer_2(self, ref, reader, tmpdir):
        with tmpdir.as_cwd():
            outfile = zarr.open_group("test-writer" + ref.ext, "a")
            with reader.Writer(outfile, n_atoms=100) as W:
                assert_equal(isinstance(W, ref.writer), True)
                assert_equal(W.n_atoms, 100)

    def test_copying(self, ref, reader):
        original = zarrtraj.ZarrTrajReader(
            ref.trajectory, convert_units=False, dt=2, time_offset=10, foo="bar"
        )
        copy = original.copy()

        assert original.format not in ("MEMORY", "CHAIN")
        assert original.convert_units is False
        assert copy.convert_units is False
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
class TestZarrTrajWriterBaseAPI(BaseWriterTest):
    """Tests ZarrTrajWriter with with synthetic trajectory."""

    @staticmethod
    @pytest.fixture()
    def ref():
        yield ZARRTRAJReference()

    def test_write_different_box(self, ref, universe, tmpdir):
        if ref.changing_dimensions:
            outfn = "write-dimensions-test" + ref.ext
            with tmpdir.as_cwd():
                outfile = zarr.open_group(outfn, mode="a")
                with ref.writer(
                    outfile, universe.atoms.n_atoms, format="ZARRTRAJ"
                ) as W:
                    for ts in universe.trajectory:
                        universe.dimensions[:3] += 1
                        W.write(universe)

                written = ref.reader(outfile)

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
        tmpdir,
    ):
        sel_str = "resid 1"
        sel = universe.select_atoms(sel_str)
        outfn = "write-selection-test." + ref.ext
        with tmpdir.as_cwd():
            outfile = zarr.open_group(outfn, mode="a")
            with ref.writer(outfile, sel.n_atoms, format="ZARRTRAJ") as W:
                for ts in universe.trajectory:
                    W.write(sel.atoms)

            copy = ref.reader(outfile)
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

    def test_write_none(self, ref, tmpdir):
        outfn = "write-none." + ref.ext
        with tmpdir.as_cwd():
            outfile = zarr.open_group(outfn, mode="a")
            with pytest.raises(TypeError):
                with ref.writer(outfile, 42, format="ZARRTRAJ") as w:
                    w.write(None)

    def test_write_not_changing_ts(self, ref, universe, tmpdir):
        outfn = "write-not-changing-ts." + ref.ext
        copy_ts = universe.trajectory.ts.copy()
        with tmpdir.as_cwd():
            outfile = zarr.open_group(outfn, mode="a")
            with ref.writer(outfile, n_atoms=5, format="ZARRTRAJ") as W:
                W.write(universe)
                assert_timestep_almost_equal(copy_ts, universe.trajectory.ts)

    def test_write_trajectory_atomgroup(self, ref, reader, universe, tmpdir):
        outfn = "write-atoms-test." + ref.ext
        with tmpdir.as_cwd():
            outfile = zarr.open_group(outfn, mode="a")
            with ref.writer(
                outfile, universe.atoms.n_atoms, format="ZARRTRAJ"
            ) as w:
                for ts in universe.trajectory:
                    w.write(universe.atoms)
            self._check_copy(outfile, ref, reader)

    def test_write_trajectory_universe(self, ref, reader, universe, tmpdir):
        outfn = "write-uni-test." + ref.ext
        with tmpdir.as_cwd():
            outfile = zarr.open_group(outfn, mode="a")
            with ref.writer(
                outfile, universe.atoms.n_atoms, format="ZARRTRAJ"
            ) as w:
                for ts in universe.trajectory:
                    w.write(universe)
            self._check_copy(outfile, ref, reader)


# Parameterize all possible tests with force_buffered
# To cheaply test cloud buffering without slow tests
@pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")
class TestZarrTrajReaderWithRealTrajectory(object):

    prec = 3

    @pytest.fixture()
    def universe(self):
        return mda.Universe(TPR_xvf, zarr.open_group(ZARRTRAJ_xvf, mode="a"))

    @pytest.fixture()
    def outgroup(self, tmpdir):
        file = str(tmpdir) + "zarrtraj-writer-test.zarrtraj"
        return zarr.open_group(file, mode="a")


@pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")
class TestZarrTrajWriterWithRealTrajectory(object):

    prec = 3

    @pytest.fixture()
    def universe(self):
        return mda.Universe(TPR_xvf, zarr.open_group(ZARRTRAJ_xvf, mode="a"))

    @pytest.fixture()
    def Writer(self):
        return zarrtraj.ZARRTRAJ.ZarrTrajWriter

    @pytest.fixture()
    def outgroup(self, tmpdir):
        file = str(tmpdir) + "zarrtraj-writer-test.zarrtraj"
        return zarr.open_group(file, mode="a")

    @pytest.mark.parametrize(
        "scalar, error, force_buffered, match",
        (
            (
                0,
                ValueError,
                False,
                "ZarrTrajWriter: no atoms in output trajectory",
            ),
            (
                0,
                ValueError,
                True,
                "ZarrTrajWriter: no atoms in output trajectory",
            ),
            (
                0.5,
                IOError,
                False,
                "ZarrTrajWriter: Timestep does not have "
                + "the correct number of atoms",
            ),
            (
                0.5,
                IOError,
                True,
                "ZarrTrajWriter: Timestep does not have "
                + "the correct number of atoms",
            ),
        ),
    )
    def test_n_atoms_errors(
        self, universe, Writer, outgroup, scalar, error, force_buffered, match
    ):
        n_atoms = universe.atoms.n_atoms * scalar
        with pytest.raises(error, match=match):
            with Writer(
                outgroup,
                n_atoms,
                n_frames=universe.trajectory.n_frames,
                force_buffered=force_buffered,
                format="ZARRTRAJ",
            ) as W:
                W.write(universe)

    def test_max_memory_too_low(self, Writer, universe, outgroup):
        with pytest.raises(ValueError, match="ZarrTrajWriter: `max_memory`"):
            with Writer(
                outgroup,
                universe.atoms.n_atoms,
                n_frames=universe.trajectory.n_frames,
                chunks=(1, universe.trajectory.n_atoms, 3),
                max_memory=get_frame_size(universe) - 1,
                force_buffered=True,
                format="ZARRTRAJ",
            ) as w:
                for ts in universe.trajectory:
                    w.write(universe)

    def test_max_memory_usage(self, Writer, universe, outgroup):
        five_framesize = get_frame_size(universe) * 5
        with Writer(
            outgroup,
            universe.atoms.n_atoms,
            n_frames=universe.trajectory.n_frames,
            chunks=(5, universe.trajectory.n_atoms, 3),
            max_memory=five_framesize,
            force_buffered=True,
            format="ZARRTRAJ",
        ) as w:
            for ts in universe.trajectory[:5]:
                w.write(universe)
            assert get_memory_usage(w) <= five_framesize
