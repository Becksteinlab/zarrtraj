import zarrtraj
import zarr
import os
import boto3
import s3fs
import MDAnalysis as mda
import numcodecs
import numpy as np

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
from numpy.testing import assert_equal
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
    obj_name = os.path.basename(file_name)
    s3_fs = s3fs.S3FileSystem()
    s3_fs.put(file_name, "zarrtraj-test-data/" + obj_name, recursive=True)

    return True


class H5MDFmtReference(BaseReference):
    """Reference synthetic trajectory that was
    copied from test_xdr.TRRReference"""

    def __init__(self, filename):
        super(H5MDFmtReference, self).__init__()
        self.trajectory = filename
        self.topology = COORDINATES_TOPOLOGY
        self.reader = zarrtraj.ZARRH5MDReader
        self.writer = zarrtraj.ZARRMDWriter
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
    ids=["s3-h5md", "local-h5md", "s3-zarrmd", "local-zarrmd"],
    indirect=True,
)
class TestH5MDFmtReaderBaseAPI(MultiframeReaderTest):
    """Tests ZarrTrajReader with with synthetic trajectory."""

    @pytest.mark.parametrize("asel", ("index 1", "index 2", "index 1 to 3"))
    def test_timeseries_asel_shape(self, reader, asel, ref):
        if ref.trajectory.startswith("s3") and ref.trajectory.endswith(
            ".zarrmd"
        ):
            pytest.skip("Duplicate key issue with moto")
        atoms = mda.Universe(reader.filename).select_atoms("index 1")
        timeseries = reader.timeseries(atoms, order="fac")
        assert timeseries.shape[0] == len(reader)
        assert timeseries.shape[1] == len(atoms)
        assert timeseries.shape[2] == 3

    def test_timeseries_empty_asel(self, reader, ref):
        if ref.trajectory.startswith("s3") and ref.trajectory.endswith(
            ".zarrmd"
        ):
            pytest.skip("Duplicate key issue with moto")
        with pytest.warns(
            UserWarning,
            match="Empty string to select atoms, empty group returned.",
        ):
            atoms = mda.Universe(reader.filename).select_atoms(None)
        with pytest.raises(ValueError, match="Timeseries requires at least"):
            reader.timeseries(asel=atoms)

    def test_timeseries_empty_atomgroup(self, reader, ref):
        if ref.trajectory.startswith("s3") and ref.trajectory.endswith(
            ".zarrmd"
        ):
            pytest.skip("Duplicate key issue with moto")
        with pytest.warns(
            UserWarning,
            match="Empty string to select atoms, empty group returned.",
        ):
            atoms = mda.Universe(reader.filename).select_atoms(None)
        with pytest.raises(ValueError, match="Timeseries requires at least"):
            reader.timeseries(atomgroup=atoms)

    def test_timeseries_asel_warns_deprecation(self, reader, ref):
        if ref.trajectory.startswith("s3") and ref.trajectory.endswith(
            ".zarrmd"
        ):
            pytest.skip("Duplicate key issue with moto")
        atoms = mda.Universe(reader.filename).select_atoms("index 1")
        with pytest.warns(DeprecationWarning, match="asel argument to"):
            timeseries = reader.timeseries(asel=atoms, order="fac")

    def test_timeseries_atomgroup(self, reader, ref):
        if ref.trajectory.startswith("s3") and ref.trajectory.endswith(
            ".zarrmd"
        ):
            pytest.skip("Duplicate key issue with moto")
        atoms = mda.Universe(reader.filename).select_atoms("index 1")
        timeseries = reader.timeseries(atomgroup=atoms, order="fac")

    def test_timeseries_atomgroup_asel_mutex(self, reader, ref):
        if ref.trajectory.startswith("s3") and ref.trajectory.endswith(
            ".zarrmd"
        ):
            pytest.skip("Duplicate key issue with moto")
        atoms = mda.Universe(reader.filename).select_atoms("index 1")
        with pytest.raises(ValueError, match="Cannot provide both"):
            timeseries = reader.timeseries(
                atomgroup=atoms, asel=atoms, order="fac"
            )


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
        for i in range(5):
            assert reader[i].time == i
            assert reader[i].data["step"] == i

        assert len(reader) == 5

    def test_explicit_offset_velocities(self, reader, true_vals):
        for i in range(3):
            assert_array_almost_equal(
                reader[i].velocities, true_vals[i].velocities
            )
        for i in range(3, 5):
            assert not reader[i].has_velocities

    def test_fixed_offset_forces(self, reader, true_vals):
        assert not reader[0].has_forces
        for i in range(1, 5):
            assert_array_almost_equal(reader[i].forces, true_vals[i].forces)

    def test_fixed_positions(self, reader, true_vals):
        for i in range(5):
            assert_array_almost_equal(
                reader[i].positions, true_vals[i].positions
            )

    def test_particle_group_observables(self, reader):
        assert reader[0].data["trajectory/obsv1"] is None
        assert reader[0].data["trajectory/obsv2"] is None
        for i in range(1, 5, 2):
            assert reader[i].data["trajectory/obsv1"] == i + 3
            assert reader[i].data["trajectory/obsv2"] == i + 3
        for i in range(0, 5, 2):
            assert reader[i].data["trajectory/obsv1"] is None
            assert reader[i].data["trajectory/obsv2"] is None

    def test_global_observables(self, reader):
        assert reader[0].data["obsv1"] is None
        assert reader[0].data["obsv2"] is None
        for i in range(1, 5, 2):
            assert reader[i].data["obsv1"] == i + 3
            assert reader[i].data["obsv2"] == i + 3
        for i in range(0, 5, 2):
            assert reader[i].data["obsv1"] is None
            assert reader[i].data["obsv2"] is None

    def test_time_independent_observables(self, reader):
        for i in range(5):
            assert_array_almost_equal(reader[i].data["obsv3"], [4, 6, 8])
            assert_array_almost_equal(
                reader[i].data["trajectory/obsv3"], [4, 6, 8]
            )


# Override all writer tests to provide n_frames kwarg
# and parameterize s3 vs disk
@pytest.mark.parametrize(
    "prefix",
    [
        "s3://zarrtraj-test-data/",
        "",
    ],
)
class TestH5MDFmtWriterBaseAPI(BaseWriterTest):

    @staticmethod
    @pytest.fixture()
    def ref():
        return H5MDFmtReference(COORDINATES_SYNTHETIC_H5MD)

    def test_write_different_box(self, ref, universe, tmpdir, prefix):
        if ref.changing_dimensions:
            outfile = prefix + "write-dimensions-test." + ref.ext
            with tmpdir.as_cwd():
                with ref.writer(
                    outfile,
                    universe.atoms.n_atoms,
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

    def test_write_trajectory_atomgroup(
        self, ref, reader, universe, tmpdir, prefix
    ):
        outfile = prefix + "write-atoms-test." + ref.ext
        with tmpdir.as_cwd():
            with ref.writer(
                outfile,
                universe.atoms.n_atoms,
            ) as w:
                for ts in universe.trajectory:
                    w.write(universe.atoms)
            self._check_copy(outfile, ref, reader)

    def test_write_trajectory_universe(
        self, ref, reader, universe, tmpdir, prefix
    ):
        outfile = prefix + "write-uni-test." + ref.ext
        with tmpdir.as_cwd():
            with ref.writer(
                outfile,
                universe.atoms.n_atoms,
            ) as w:
                for ts in universe.trajectory:
                    w.write(universe)
            self._check_copy(outfile, ref, reader)

    def test_write_selection(
        self,
        ref,
        reader,
        universe,
        u_no_resnames,
        u_no_resids,
        u_no_names,
        tmpdir,
        prefix,
    ):
        sel_str = "resid 1"
        sel = universe.select_atoms(sel_str)
        outfile = prefix + "write-selection-test." + ref.ext

        with tmpdir.as_cwd():
            with ref.writer(
                outfile,
                sel.n_atoms,
            ) as W:
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

    def test_write_not_changing_ts(self, ref, universe, tmpdir, prefix):
        outfile = prefix + "write-not-changing-ts." + ref.ext

        copy_ts = universe.trajectory.ts.copy()
        with tmpdir.as_cwd():
            with ref.writer(
                outfile,
                n_atoms=5,
            ) as W:
                W.write(universe)
                assert_timestep_almost_equal(copy_ts, universe.trajectory.ts)

    def test_write_none(self, ref, tmpdir, prefix):
        outfile = prefix + "write-none." + ref.ext
        with tmpdir.as_cwd():
            with pytest.raises(TypeError):
                with ref.writer(outfile, 42) as w:
                    w.write(None)

    def test_no_container(self, ref, tmpdir, prefix):
        with tmpdir.as_cwd():
            if ref.container_format:
                ref.writer("foo")
            else:
                with pytest.raises(TypeError):
                    ref.writer("foo")


class TestH5MDFmtWriterNumcodecs:
    @pytest.fixture()
    def universe(self):
        return mda.Universe(TPR_xvf, H5MD_xvf)

    def test_write_compressor_filters(self, universe, tmpdir):
        outfile = "write-compressor-test.zarrmd"
        with tmpdir.as_cwd():
            with mda.Writer(
                outfile,
                universe.atoms.n_atoms,
                compressor=numcodecs.Blosc(cname="zstd", clevel=7),
                precision=3,
            ) as w:
                for ts in universe.trajectory:
                    w.write(universe)
            written = zarr.open_group(outfile, mode="r")

            particle_group_elems = [
                "position",
                "velocity",
                "force",
                "box/edges",
            ]
            particle_group_elems = [
                "/particles/trajectory/" + elem for elem in particle_group_elems
            ]
            obsv_elems = ["/observables/trajectory/lambda"]
            particle_group_elems.extend(obsv_elems)
            for h5mdelem_path in particle_group_elems:
                assert written[h5mdelem_path][
                    "value"
                ].compressor == numcodecs.Blosc(cname="zstd", clevel=7)
                assert written[h5mdelem_path]["value"].filters == [
                    numcodecs.Quantize(
                        3, dtype=written[h5mdelem_path]["value"].dtype
                    )
                ]
                assert written[h5mdelem_path][
                    "time"
                ].compressor == numcodecs.Blosc(cname="zstd", clevel=7)
                assert written[h5mdelem_path]["time"].filters == [
                    numcodecs.Quantize(3, np.float32)
                ]
                assert written[h5mdelem_path][
                    "step"
                ].compressor == numcodecs.Blosc(cname="zstd", clevel=7)


class TestH5MDFmtWriterNFrames:
    @pytest.fixture()
    def universe(self):
        return mda.Universe(TPR_xvf, H5MD_xvf)

    def test_write_below_n_frames(self, universe, tmpdir):
        """Writing below n_frames should issue a warning but still write
        the correct number of frames"""

        with tmpdir.as_cwd():
            with pytest.warns(
                RuntimeWarning,
            ):
                with mda.Writer(
                    "foo.zarrmd",
                    universe.atoms.n_atoms,
                    n_frames=2,
                ) as w:
                    w.write(universe)

            written = mda.Universe(TPR_xvf, "foo.zarrmd")
            for i in range(len(written.trajectory)):
                assert_timestep_almost_equal(
                    written.trajectory[i], universe.trajectory[i]
                )

            assert len(written.trajectory) == 1

    def test_write_above_n_frames(self, universe, tmpdir):
        """Writing above n_frames should issue a warning but still write
        the correct number of frames"""
        with tmpdir.as_cwd():
            with pytest.warns(
                RuntimeWarning,
            ):
                with mda.Writer(
                    "foo.zarrmd",
                    universe.atoms.n_atoms,
                    n_frames=2,
                ) as w:
                    for ts in universe.trajectory:
                        w.write(universe)

            written = mda.Universe(TPR_xvf, "foo.zarrmd")
            for i in range(len(written.trajectory)):
                assert_timestep_almost_equal(
                    written.trajectory[i], universe.trajectory[i]
                )

            assert len(written.trajectory) == 3
