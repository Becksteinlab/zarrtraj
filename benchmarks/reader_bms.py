from zarrtraj import *
import MDAnalysis as mda

from zarr.storage import DirectoryStore, LRUStoreCache
import MDAnalysis.analysis.rms as rms
from MDAnalysis.coordinates.H5MD import H5MDReader
import zarr
import h5py
import dask.array as da

import os

from dask.distributed import Client, LocalCluster


"""
1. Activate the devtools/asv_env.yaml environment

2. Make sure to set the BENCHMARK_DATA_DIR to wherever local yiip files are stored

3. To run, use:

Development:

    asv run -q -v -e <commit> > bm.log &

Full run:

    asv run -v -e <commit> > bm.log &

4. To publish, use


"""

BENCHMARK_DATA_DIR = os.getenv("BENCHMARK_DATA_DIR")
os.environ["S3_REGION_NAME"] = "us-west-1"
os.environ["AWS_PROFILE"] = "sample_profile"


s3_files = [
    "s3://zarrtraj-test-data/yiip_aligned_compressed.zarrmd",
    "s3://zarrtraj-test-data/yiip_aligned_uncompressed.zarrmd",
    "s3://zarrtraj-test-data/yiip_aligned_compressed.h5md",
    "s3://zarrtraj-test-data/yiip_aligned_uncompressed.h5md",
]
local_files = [
    f"{BENCHMARK_DATA_DIR}/yiip_aligned_compressed.zarrmd",
    f"{BENCHMARK_DATA_DIR}/yiip_aligned_uncompressed.zarrmd",
    f"{BENCHMARK_DATA_DIR}/yiip_aligned_compressed.h5md",
    f"{BENCHMARK_DATA_DIR}/yiip_aligned_uncompressed.h5md",
]

h5md_files = [
    f"{BENCHMARK_DATA_DIR}/yiip_aligned_compressed.h5md",
    f"{BENCHMARK_DATA_DIR}/yiip_aligned_uncompressed.h5md",
]

s3_zarrmd_files = [
    "s3://zarrtraj-test-data/yiip_aligned_compressed.zarrmd",
    "s3://zarrtraj-test-data/yiip_aligned_uncompressed.zarrmd",
]

local_zarrmd_files = [
    f"{BENCHMARK_DATA_DIR}/yiip_aligned_compressed.zarrmd",
    f"{BENCHMARK_DATA_DIR}/yiip_aligned_uncompressed.zarrmd",
]


def dask_rmsf(positions):
    mean_positions = positions.mean(axis=0)
    subtracted_positions = positions - mean_positions
    squared_deviations = subtracted_positions**2
    avg_squared_deviations = squared_deviations.mean(axis=0)
    sqrt_avg_squared_deviations = da.sqrt(avg_squared_deviations)
    return da.sqrt((sqrt_avg_squared_deviations**2).sum(axis=1))


class ZARRH5MDDiskStrideTime(object):
    """Benchmarks for zarrmd and h5md file striding using local files."""

    params = local_files
    param_names = ["filename"]
    timeout = 2400.0

    def setup(self, filename):
        self.reader_object = ZARRH5MDReader(filename)

    def time_strides(self, filename):
        """Benchmark striding over full trajectory"""
        for ts in self.reader_object:
            pass

    def teardown(self, filename):
        del self.reader_object


class ZARRH5MDS3StrideTime(object):
    """Benchmarks for zarrmd and h5md file striding using local files."""

    params = s3_files
    param_names = ["filename"]
    timeout = 2400.0

    def setup(self, filename):
        self.reader_object = ZARRH5MDReader(filename)

    def time_strides(self, filename):
        """Benchmark striding over full trajectory"""
        for ts in self.reader_object:
            pass

    def teardown(self, filename):
        del self.reader_object


class H5MDReadersDiskStrideTime(object):
    """Benchmarks for zarrmd and h5md file striding using local files."""

    params = (h5md_files, [ZARRH5MDReader, H5MDReader])
    param_names = ["filename", "reader"]
    timeout = 2400.0

    def setup(self, filename, reader):
        self.reader_object = reader(filename)

    def time_strides(self, filename, reader):
        """Benchmark striding over full trajectory"""
        for ts in self.reader_object:
            pass

    def teardown(self, filename, reader):
        del self.reader_object


class H5MDFmtDiskRMSFTime(object):

    params = (local_zarrmd_files, ["dask", "mda"])
    param_names = ["filename", "method"]
    timeout = 2400.0

    def setup(self, filename, method):
        if method == "dask":
            self.positions = da.from_array(
                zarr.open_group(filename)[
                    "/particles/trajectory/position/value"
                ]
            )

        elif method == "mda":
            self.universe = mda.Universe(
                f"{BENCHMARK_DATA_DIR}/yiip_equilibrium/YiiP_system.pdb",
                filename,
            )

    def time_rmsf(self, filename, method):
        """Benchmark striding over full trajectory"""
        if method == "mda":
            rms.RMSF(self.universe.atoms).run()
        elif method == "dask":
            rmsf = dask_rmsf(self.positions)
            rmsf.compute()

    def teardown(self, filename, method):
        if hasattr(self, "positions"):
            del self.positions
        if hasattr(self, "universe"):
            del self.universe


class H5MDFmtAWSRMSFTime(object):

    params = (s3_zarrmd_files, ["dask", "mda"])
    param_names = ["filename", "method"]
    timeout = 2400.0

    def setup(self, filename, method):
        if method == "dask":
            self.positions = da.from_array(
                zarr.open_group(filename)[
                    "/particles/trajectory/position/value"
                ]
            )

        elif method == "mda":
            self.universe = mda.Universe(
                f"{BENCHMARK_DATA_DIR}/yiip_equilibrium/YiiP_system.pdb",
                filename,
            )

    def time_rmsf(self, filename, method):
        """Benchmark striding over full trajectory"""
        if method == "mda":
            rms.RMSF(self.universe.atoms).run()
        elif method == "dask":
            rmsf = dask_rmsf(self.positions)
            rmsf.compute()

    def teardown(self, filename, method):
        if hasattr(self, "positions"):
            del self.positions
        if hasattr(self, "universe"):
            del self.universe
