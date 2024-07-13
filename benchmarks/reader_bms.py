from zarrtraj import *
import MDAnalysis as mda

from zarr.storage import DirectoryStore, LRUStoreCache
import MDAnalysis.analysis.rms as rms
from MDAnalysis.coordinates.H5MD import H5MDReader
import zarr
import h5py
import dask as da

import os


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

    def setup(self, filename):
        self.reader_object = ZARRH5MDReader(filename)

    def time_strides(self, filename):
        """Benchmark striding over full trajectory"""
        for ts in self.reader_object:
            pass


class ZARRH5MDS3StrideTime(object):
    """Benchmarks for zarrmd and h5md file striding using local files."""

    params = s3_files
    param_names = ["filename"]

    def setup(self, filename):
        self.reader_object = ZARRH5MDReader(filename)

    def time_strides(self, filename):
        """Benchmark striding over full trajectory"""
        for ts in self.reader_object:
            pass


class H5MDReadersDiskStrideTime(object):
    """Benchmarks for zarrmd and h5md file striding using local files."""

    params = (h5md_files, [ZARRH5MDReader, H5MDReader])
    param_names = ["filename", "reader"]

    def setup(self, filename, reader):
        self.reader_object = reader(filename)

    def time_strides(self, filename, reader):
        """Benchmark striding over full trajectory"""
        for ts in self.reader_object:
            pass


class H5MDFmtDiskRSMFTime(object):
    """Benchmarks for zarrtraj file striding."""

    params = (local_files, ["dask", "mda"])
    param_names = ["filename", "method"]

    def setup(self, filename, method):
        if method == "dask":
            if filename.endswith(".h5md"):
                self.positions = da.from_array(
                    h5py.File(filename)["/particles/trajectory/position/value"]
                )

            elif filename.endswith("zarrmd"):
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


class H5MDFmtAWSRSMFTime(object):
    """Benchmarks for zarrtraj file striding."""

    params = (s3_files, ["dask", "mda"])
    param_names = ["filename", "method"]

    def setup(self, filename, method):
        if method == "dask":
            if filename.endswith(".h5md"):
                self.positions = da.from_array(
                    h5py.File(filename)["/particles/trajectory/position/value"]
                )

            elif filename.endswith(".zarrmd"):
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
            rms.RMSF(self.universe.atoms)
        elif method == "dask":
            rmsf = dask_rmsf(self.positions)
            rmsf.compute()
