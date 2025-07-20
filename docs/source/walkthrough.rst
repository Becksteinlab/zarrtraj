.. _walkthrough:

Walkthrough
===========

This walkthrough will guide you through the process of writing and then reading H5MD-formatted trajectories from cloud storage using 
AWS S3 as an example. To learn more about reading and writing trajectories from different cloud storage providers, 
including Google Cloud and Azure, see the :ref:`API documentation <api>`.

We will use the `YiiP 9ns trajectory from MDAnalysisData <https://www.mdanalysis.org/MDAnalysisData/yiip_equilibrium.html>`_
as an example trajectory for the walkthrough.

.. note:: 
    In examples that read or write the :ref:`ZarrMD format <zarrmd>`, ``zarrtraj`` is imported even though 
    its namespace is not explicitly called. This is because MDAnalysis uses `meta classes for registering reader and writer formats <https://github.com/MDAnalysis/mdanalysis/blob/d412c9a9a56312c1bd4e33e6dd3afc4cec7783ca/package/MDAnalysis/coordinates/base.py>`_,
    so calling ``import zarrtraj`` allows MDAnalysis to use the classes provided by this package.

Writing trajectories to cloud storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, we show how to upload both H5MD files and ZarrMD files to an S3 bucket.

Setup: Creating an AWS S3 bucket
################################

First, create an AWS S3 bucket. This requires that an S3 Bucket is setup and configured for 
write access using the credentials stored in "sample_profile" (for the examples shown here, but a profile can be named arbitrarily). 
If you've never configured an S3 Bucket before, see
`this guide <https://docs.aws.amazon.com/AmazonS3/latest/userguide/creating-bucket.html>`_. You can setup a profile to easily manage AWS
credentials using `this VSCode extension <https://marketplace.visualstudio.com/items?itemName=AmazonWebServices.aws-toolkit-vscode>`_.
Here is a sample profile (stored in ~/.aws/credentials) where 
`the key is an access key associated with a user that has read and write permissions for the bucket 
<https://stackoverflow.com/questions/50802319/create-a-single-iam-user-to-access-only-specific-s3-bucket>`_::

    [sample_profile]
    aws_access_key_id = <key>


Writing trajectories directly from MDAnalysis into a ZarrMD file in an S3 Bucket (preferred method)
####################################################################################################

Using the created credentials with read/write access, you can write a trajectory
into your bucket.

You can change the stored precision of floating point values in the trajectory file with the optional
``precision`` kwarg and pass in any ```numcodecs.Codec``` compressor with the optional
``compressor`` kwarg. See `numcodecs <https://numcodecs.readthedocs.io/en/stable/>`_
for more on the available compressors.

Chunking is automatically determined for all datasets to be optimized for
cloud storage reading speed and is not configurable by the user::

    import zarrtraj
    import MDAnalysis as mda
    import MDAnalysisData
    import numcodecs
    import os

    os.environ["AWS_PROFILE"] = "sample_profile"
    os.environ["AWS_REGION"] = "us-west-1"

    if __name__ == "__main__":
        dataset = MDAnalysisData.fetch_yiip_equilibrium_short()

        u = mda.Universe(dataset.topology, dataset.trajectory)

        with mda.Writer(
            "s3://sample-bucket-name/YiiP_system_9ns_center.zarrmd",
            n_atoms=u.atoms.n_atoms,
            precision=3,
            compressor=numcodecs.Blosc(cname="zstd", clevel=9),
        ) as W:
            for ts in u.trajectory:
                W.write(u.atoms)


Uploading an H5MD file to an S3 bucket
######################################

.. note:: 
    Uploading H5MD trajectories programmatically to AWS S3 requires the `Boto3 package <https://github.com/boto/boto3>`_ to be installed.
    This won't be installed by default with Zarrtraj because we recommend writing directly to S3 using the  :ref:`ZarrMD format <zarrmd>`

MDAnalysis can write a trajectory from
`any of its supported formats into H5MD <https://docs.mdanalysis.org/stable/documentation_pages/coordinates/H5MD.html>`_. 
Because the ``H5MDWriter`` (provided by MDAnalysis, not this package) does not automatically determine optimal chunks, we 
recommend using the ```chunks``` kwarg with the MDAnalysis ``H5MDWriter`` with a value that yields ~8-16MB chunks of data for best S3 performance.
Once written locally, you can upload the trajectory to S3 programatically::

    import MDAnalysisData
    import MDAnalysis as mda
    import os
    import boto3

    os.environ["AWS_PROFILE"] = "sample_profile"
    # This is the AWS region where the bucket is located
    os.environ["AWS_REGION"] = "us-west-1"

    def upload_h5md_file(bucket_name, file_name):
        s3_client = boto3.client("s3")
        obj_name = os.path.basename(file_name)

        response = s3_client.upload_file(file_name, bucket_name, obj_name)

    if __name__ == "__main__":
        dataset = MDAnalysisData.fetch_yiip_equilibrium_short()

        u = mda.Universe(dataset.topology, dataset.trajectory)

        with mda.Writer(
            "YiiP_system_9ns_center.h5md",
            n_atoms=u.atoms.n_atoms,
            # (111815 atoms * 4 bytes per float * 3 (xyz)) = ~1.34 MB per frame
            # 8 frames per chunk to reach goal of 8-12 MB per chunk
            chunks=(8, u.atoms.n_atoms, 3),
        ) as W:
            for ts in u.trajectory:
                W.write(u.atoms)

        upload_h5md_file(
            "sample-bucket-name",
            "YiiP_system_9ns_center.h5md",
        )

You can also upload the H5MD file directly using the AWS web interface by navigating to S3, the bucket name, and pressing
"upload".

Reading your H5MD file
######################

After the file is uploaded, you can use the same credentials to stream the file into MDAnalysis::

    import zarrtraj
    import MDAnalysis as mda
    import MDAnalysisData
    import os

    os.environ["AWS_PROFILE"] = "sample_profile"
    os.environ["AWS_REGION"] = "us-west-1"

    dataset = MDAnalysisData.yiip_equilibrium.fetch_yiip_equilibrium_short()
    # here, we show the .zarrmd file being read, but the .h5md file could be read identically
    u = mda.Universe(dataset.topology, "s3://sample-bucket-name/YiiP_system_9ns_center.zarrmd")
    protein = u.select_atoms("protein")
    for ts in u.trajectory[::100]:
        print(f"{ts.frame}, {ts.time}, {protein.center_of_mass()}")
        

If you have additional questions, please don't hesitate to open a discussion on the `zarrtarj github <https://github.com/Becksteinlab/zarrtraj>`_.
The `MDAnalysis discord <https://discord.com/channels/807348386012987462/>`_ is also a 
great resource for asking questions and getting involved in MDAnalysis.