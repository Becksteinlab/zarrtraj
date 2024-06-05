import re
import pathlib
import fsspec


def get_protocol(url: str) -> str:
    parts = re.split(r"(\:\:|\://)", url, maxsplit=1)
    if len(parts) > 1:
        return parts[0]
    return "file"


def get_extension(url: str) -> str:
    return pathlib.Path(url).suffix


def get_h5_zarr_mapping(
    url: str,
    protocol: str,
    so: dict,
) -> str:
    try:
        import kerchunk
    except ImportError:
        raise ImportError(
            "Please install kerchunk to read H5MD files"
        ) from None

    with fsspec.open(url, **so) as inf:
        h5chunks = kerchunk.hdf.SingleHdf5ToZarr(inf, url, inline_threshold=100)
        fo = h5chunks.translate()

    if protocol == "s3":
        fs = fsspec.filesystem(
            "reference",
            fo=fo,
            remote_protocol="s3",
            remote_options=so,
            skip_instance_cache=True,
        )
    elif protocol == "file":
        fs = fsspec.filesystem(
            "reference",
            fo=fo,
            skip_instance_cache=True,
        )
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    return fs.get_mapper("")
