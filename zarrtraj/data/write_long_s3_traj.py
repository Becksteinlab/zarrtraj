import zarrtraj
import zarr
from numcodecs import Blosc, Quantize
import MDAnalysis as mda
import MDAnalysisData

yiip = MDAnalysisData.yiip_equilibrium.fetch_yiip_equilibrium_long(
    data_home="notebook_data_tmp"
)
# 901 frames of 111815 atoms
# each frame is 1341780 bytes (1.34178 mB)
#
uLong = mda.Universe(yiip.topology, yiip.trajectory)

storage_options = {
    "anon": False,
    "s3": {
        "profile": "sample_profile",
        "client_kwargs": {"region_name": "us-west-1"},
    },
}

compressor_lvl = [0, 1, 9]
filters_list = ["all", 3]
# approx 1.3 mb, 13 mB, 130mB
chunks_frames = [1, 10, 100]

for c in compressor_lvl:
    for f in filters_list:
        for ch in chunks_frames:
            if f == 3:
                filters = [Quantize(digits=3, dtype="f4")]
            else:
                filters = None
            z = f"s3://zarrtraj-test-data/long_{c}_{f}_{ch}.zarrtraj"

            with mda.Writer(
                z,
                n_atoms=uLong.trajectory.n_atoms,
                n_frames=uLong.trajectory.n_frames,
                force_buffered=True,
                compressor=Blosc(cname="zstd", clevel=c),
                filters=filters,
                chunks=(ch, uLong.trajectory.n_atoms, 3),
                storage_options=storage_options,
                max_memory=2**28,
            ) as w:
                for ts in uLong.trajectory:
                    w.write(uLong)
