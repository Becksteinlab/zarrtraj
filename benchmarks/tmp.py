import zarrtraj
import MDAnalysisData
import MDAnalysis as mda
import os

yiip = MDAnalysisData.yiip_equilibrium.fetch_yiip_equilibrium_long()


# os.environ["S3_REGION_NAME"] = "us-west-1"
# os.environ["AWS_PROFILE"] = "sample_profile"

storage_options = {
    # "cache_type": "readahead",
    "anon": False,
    "profile": "sample_profile",
    "client_kwargs": {
        "region_name": "us-west-1",
    },
}

u = mda.Universe(
    yiip.topology,
    "s3://zarrtraj-test-data/short_0_3_1.zarrtraj",
    storage_options=storage_options,
)

print(u.trajectory[0])
