Full walkthrough
================

Reading H5MD trajectories from cloud storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Uploading your H5MD file
########################

First, upload your H5MD trajectories to an AWS S3 bucket. This requires that an S3 Bucket is setup and configured for 
write access using the credentials stored in "sample_profile" If you've never configured an S3 Bucket before, see
`this guide <https://docs.aws.amazon.com/AmazonS3/latest/userguide/creating-bucket.html>`_. You can setup a profile to easily manage AWS
credentials using `this VSCode extension <https://marketplace.visualstudio.com/items?itemName=AmazonWebServices.aws-toolkit-vscode>`_.
Here is a sample profile (stored in ~/.aws/credentials) where 
`the key is an access key associated with a user that has read and write permissions for the bucket 
<https://stackoverflow.com/questions/50802319/create-a-single-iam-user-to-access-only-specific-s3-bucket>`_::

    [sample_profile]
    aws_access_key_id = <key>

MDAnalysis can write a trajectory from
`any of its supported formats into H5MD <https://docs.mdanalysis.org/stable/documentation_pages/coordinates/H5MD.html>`_. We 
recommend using the `chunks` kwarg with the MDAnalysis H5MDWriter with a value that yields ~8-16MB chunks of data for best S3 performance.
Once written locally, you can upload the trajectory to S3 programatically::

    import os
    from botocore.exceptions import ClientError
    import boto3
    import logging

    os.environ["AWS_PROFILE"] = "sample_profile"
    # This is the AWS region the bucket is stored in
    os.environ["AWS_REGION"] = "us-west-1"

    def upload_h5md_file(bucket_name, file_name):
        s3_client = boto3.client("s3")
        obj_name = os.path.basename(file_name)
        try:
            response = s3_client.upload_file(
                file_name, bucket_name, obj_name
            )
        except ClientError as e:
            logging.error(e)
            return False
        return True

    if __name__ == "__main__":
        # Using test H5MD file from the zarrtraj repo
        upload_h5md_file("sample-bucket-name", "zarrtraj/data/COORDINATES_SYNTHETIC_H5MD.h5md")

You can also upload the H5MD file directly using the AWS web interface by navigating to S3, the bucket name, and pressing
"upload".

Reading your H5MD file
######################

After the file is uploaded, you can use the same credentials to stream the file into MDAnalysis::

    import zarrtraj
    import MDAnalysis as mda
    # This sample topology requires installing MDAnalysisTests
    from MDAnalysisTests.datafiles import COORDINATES_TOPOLOGY
    import os

    os.environ["AWS_PROFILE"] = "sample_profile"
    os.environ["AWS_REGION"] = "us-west-1"

    u = mda.Universe(COORDINATES_TOPOLOGY, "s3://sample-bucket-name/COORDINATES_SYNTHETIC_H5MD.h5md")
    for ts in u.trajectory:
        pass

You can follow this same process for reading `.zarrmd` files with the added advantage
that Zarrtarj can write `.zarrmd` files directly into an S3 bucket.

Writing trajectories from MDAnalysis into a zarrmd file in an S3 Bucket
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the same credentials with read/write access, you can write a trajectory
into your bucket. Note that the `n_frames` kwarg is required unlike 
with other MDAnalysis writers, though this may not be the case in future releases.

You can change the stored precision of floating point values in the file with the optional
`precision` kwarg and pass in any `numcodecs.Codec` compressor with the optional
`compressor` kwarg. See [numcodecs](https://numcodecs.readthedocs.io/en/stable/)
for more on the available compressors.

Chunking is automatically determined for all datasets to be optimized for
cloud storage and is not configurable by the user. 
Initial benchmarks show this chunking strategy is effective for disk storage as well.::

    import zarrtraj
    import MDAnalysis as mda
    from MDAnalysisTests.datafiles import PSF, DCD
    import numcodecs
    import os

    os.environ["AWS_PROFILE"] = "sample_profile"
    os.environ["AWS_REGION"] = "us-west-1"

    u = mda.Universe(PSF, DCD)
    with mda.Writer("s3://sample-bucket-name/test.zarrmd", 
                    n_atoms=u.trajectory.n_atoms, 
                    n_frames=u.trajectory.n_frames,
                    # Not required
                    precision=3,
                    compressor=numcodecs.Blosc(cname="zstd", clevel=9)) as W:
                    for ts in u.trajectory:
                        W.write(u.atoms)

If you have additional questions, please don't hesitate to open a discussion on the `zarrtarj github <https://github.com/Becksteinlab/zarrtraj>`_.
The `MDAnalysis discord <https://discord.com/channels/807348386012987462/>`_ is also a 
great resource for asking questions and getting involved in MDAnalysis.