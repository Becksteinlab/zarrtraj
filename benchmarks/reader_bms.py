from zarrtraj import *

# from asv_runner.benchmarks.mark import skip_for_params
from zarr.storage import DirectoryStore, LRUStoreCache
import MDAnalysis.analysis.rms as rms
from MDAnalysis.coordinates.H5MD import H5MDReader

import os

"""
Note: while h5md files are chunked at (1, n_atoms, 3), zarr files
are chunked with as many frames as can fit in 12MB

"""

BENCHMARK_DATA_DIR = os.getenv("BENCHMARK_DATA_DIR")
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

os.environ["S3_REGION_NAME"] = "us-west-1"
os.environ["AWS_PROFILE"] = "sample_profile"


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

    params = (local_files, ""
    param_names = ["filename"]

    def setup(self, filename):
        self.reader_object = ZARRH5MDReader(filename)

    def time_strides(self, filename):
        """Benchmark striding over full trajectory"""
        for ts in self.reader_object:
            pass

    # def time_RMSD(self, compressor_level, filter_precision, chunk_frames):
    #    """Benchmark RMSF calculation"""
    #    R = rms.RMSD(
    #        self.universe,
    #        self.universe,
    #        select="backbone",
    #        ref_frame=0,
    #    ).run()


class RawZarrReadBenchmarks(object):
    timeout = 86400
    params = (
        [0, 1, 9],
        ["all", 3],
        [1, 10, 100],
    )

    param_names = [
        "compressor_level",
        "filter_precision",
        "chunk_frames",
    ]

    def setup(self, compressor_level, filter_precision, chunk_frames):
        self.traj_file = f"s3://zarrtraj-test-data/long_{compressor_level}_{filter_precision}_{chunk_frames}.zarrtraj"
        store = zarr.storage.FSStore(url=self.traj_file, mode="r")
        # For consistency with zarrtraj defaults, use 256MB LRUCache store
        cache = zarr.storage.LRUStoreCache(store, max_size=2**28)
        self.zarr_group = zarr.open_group(store=cache, mode="r")
