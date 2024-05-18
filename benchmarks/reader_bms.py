from zarrtraj import *

# from asv_runner.benchmarks.mark import skip_for_params
from zarr.storage import DirectoryStore, LRUStoreCache
import MDAnalysis.analysis.rms as rms

import os

BENCHMARK_DATA_DIR = os.getenv("BENCHMARK_DATA_DIR")

os.environ["S3_REGION_NAME"] = "us-west-1"
os.environ["AWS_PROFILE"] = "sample_profile"


class TrajReaderDiskBenchmarks(object):
    """Benchmarks for zarrtraj file striding."""

    params = (
        [0, 1, 9],
        ["all", 3],
        [1, 10, 50],
    )
    param_names = [
        "compressor_level",
        "filter_precision",
        "chunk_frames",
    ]

    def setup(
        self,
        compressor_level,
        filter_precision,
        chunk_frames,
    ):
        self.traj_file = f"{BENCHMARK_DATA_DIR}/short_{compressor_level}_{filter_precision}_{chunk_frames}.zarrtraj"
        self.reader_object = ZarrTrajReader(self.traj_file)

    def time_strides(
        self,
        compressor_level,
        filter_precision,
        chunk_frames,
    ):
        """Benchmark striding over full trajectory"""
        for ts in self.reader_object:
            pass


class TrajReaderAWSBenchmarks(object):
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
        self.traj_file = f"s3://zarrtraj-test-data/short_{compressor_level}_{filter_precision}_{chunk_frames}.zarrtraj"
        self.reader_object = ZarrTrajReader(
            self.traj_file,
        )
        self.universe = mda.Universe(
            f"{BENCHMARK_DATA_DIR}/YiiP_system.pdb", self.traj_file
        )

    def time_strides(self, compressor_level, filter_precision, chunk_frames):
        """Benchmark striding over full trajectory"""
        for ts in self.reader_object:
            pass

    def time_RMSD(self, compressor_level, filter_precision, chunk_frames):
        """Benchmark RMSF calculation"""
        R = rms.RMSD(
            self.universe,
            self.universe,
            select="backbone",
            ref_frame=0,
        ).run()
