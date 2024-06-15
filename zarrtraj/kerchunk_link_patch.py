import h5py
from zarr.meta import encode_fill_value
import base64
from kerchunk import hdf

from kerchunk.utils import (
    _encode_for_JSON,
)

from kerchunk.codecs import FillStringsCodec

import logging
from fsspec.implementations.reference import LazyReferenceMapper
from typing import Union
import numpy as np
import numcodecs


lggr = logging.getLogger("h5-to-zarr")


class SingleHdf5ToZarrPatched(hdf.SingleHdf5ToZarr):
    def translate(self, preserve_links=False):
        """Translate content of one HDF5 file into Zarr storage format.

        This method is the main entry point to execute the workflow, and
        returns a "reference" structure to be used with zarr/kerchunk

        No data is copied out of the HDF5 file.

        Parameters
        ----------
        preserve_links : bool, optional
            If True, preserve hard and soft links in the HDF5 file. Default is
            False.

        Returns
        -------
        dict
            Dictionary containing reference structure.
        """
        lggr.debug("Translation begins")
        self._transfer_attrs(self._h5f, self._zroot)

        self._preserve_links = preserve_links
        if self._preserve_links:
            self._h5f.visititems_links(self._translator)
        else:
            self._h5f.visititems(self._translator)
        if self.spec < 1:
            return self.store
        elif isinstance(self.store, LazyReferenceMapper):
            self.store.flush()
            return self.store
        else:
            store = _encode_for_JSON(self.store)
            return {"version": 1, "refs": store}

    def _translator(
        self,
        name: str,
        h5obj: Union[
            h5py.Dataset,
            h5py.Group,
            h5py.SoftLink,
            h5py.HardLink,
            h5py.ExternalLink,
        ],
    ):
        """Produce Zarr metadata for all groups and datasets in the HDF5 file."""
        try:  # method must not raise exception
            kwargs = {}

            if isinstance(h5obj, h5py.SoftLink) or isinstance(
                h5obj, h5py.HardLink
            ):
                h5obj = self._h5f[name]

            if isinstance(h5obj, h5py.Dataset):
                lggr.debug(f"HDF5 dataset: {h5obj.name}")
                lggr.debug(f"HDF5 compression: {h5obj.compression}")
                if h5obj.id.get_create_plist().get_layout() == h5py.h5d.COMPACT:
                    # Only do if h5obj.nbytes < self.inline??
                    kwargs["data"] = h5obj[:]
                    filters = []
                else:
                    filters = self._decode_filters(h5obj)
                dt = None
                # Get storage info of this HDF5 dataset...
                cinfo = self._storage_info(h5obj)

                if "data" in kwargs:
                    fill = None
                else:
                    # encodings
                    if h5obj.dtype.kind in "US":
                        fill = h5obj.fillvalue or " "  # cannot be None
                    elif h5obj.dtype.kind == "O":
                        if self.vlen == "embed":
                            if np.isscalar(h5obj):
                                out = str(h5obj)
                            elif h5obj.ndim == 0:
                                out = np.array(h5obj).tolist().decode()
                            else:
                                out = h5obj[:]
                                out2 = out.ravel()
                                for i, val in enumerate(out2):
                                    if isinstance(val, bytes):
                                        out2[i] = val.decode()
                                    elif isinstance(val, str):
                                        out2[i] = val
                                    elif isinstance(val, h5py.h5r.Reference):
                                        # TODO: recursively recreate references
                                        out2[i] = None
                                    else:
                                        out2[i] = [
                                            (
                                                v.decode()
                                                if isinstance(v, bytes)
                                                else v
                                            )
                                            for v in val
                                        ]
                            kwargs["data"] = out
                            kwargs["object_codec"] = numcodecs.JSON()
                            fill = None
                        elif self.vlen == "null":
                            dt = "O"
                            kwargs["object_codec"] = FillStringsCodec(
                                dtype="S16"
                            )
                            fill = " "
                        elif self.vlen == "leave":
                            dt = "S16"
                            fill = " "
                        elif self.vlen == "encode":
                            assert len(cinfo) == 1
                            v = list(cinfo.values())[0]
                            data = super()._read_block(
                                self.input_file, v["offset"], v["size"]
                            )
                            indexes = np.frombuffer(data, dtype="S16")
                            labels = h5obj[:]
                            mapping = {
                                index.decode(): label.decode()
                                for index, label in zip(indexes, labels)
                            }
                            kwargs["object_codec"] = FillStringsCodec(
                                dtype="S16", id_map=mapping
                            )
                            fill = " "
                        else:
                            raise NotImplementedError
                    elif hdf._is_netcdf_datetime(
                        h5obj
                    ) or hdf._is_netcdf_variable(h5obj):
                        fill = None
                    else:
                        fill = h5obj.fillvalue
                    if h5obj.dtype.kind == "V":
                        fill = None
                        if self.vlen == "encode":
                            assert len(cinfo) == 1
                            v = list(cinfo.values())[0]
                            dt = [
                                (
                                    v,
                                    (
                                        "S16"
                                        if h5obj.dtype[v].kind == "O"
                                        else str(h5obj.dtype[v])
                                    ),
                                )
                                for v in h5obj.dtype.names
                            ]
                            data = super()._read_block(
                                self.input_file, v["offset"], v["size"]
                            )
                            labels = h5obj[:]
                            arr = np.frombuffer(data, dtype=dt)
                            mapping = {}
                            for field in labels.dtype.names:
                                if labels[field].dtype == "O":
                                    mapping.update(
                                        {
                                            index.decode(): label.decode()
                                            for index, label in zip(
                                                arr[field], labels[field]
                                            )
                                        }
                                    )
                            kwargs["object_codec"] = FillStringsCodec(
                                dtype=str(dt), id_map=mapping
                            )
                            dt = [
                                (
                                    v,
                                    (
                                        "O"
                                        if h5obj.dtype[v].kind == "O"
                                        else str(h5obj.dtype[v])
                                    ),
                                )
                                for v in h5obj.dtype.names
                            ]
                        elif self.vlen == "null":
                            dt = [
                                (
                                    v,
                                    (
                                        "S16"
                                        if h5obj.dtype[v].kind == "O"
                                        else str(h5obj.dtype[v])
                                    ),
                                )
                                for v in h5obj.dtype.names
                            ]
                            kwargs["object_codec"] = FillStringsCodec(
                                dtype=str(dt)
                            )
                            dt = [
                                (
                                    v,
                                    (
                                        "O"
                                        if h5obj.dtype[v].kind == "O"
                                        else str(h5obj.dtype[v])
                                    ),
                                )
                                for v in h5obj.dtype.names
                            ]
                        elif self.vlen == "leave":
                            dt = [
                                (
                                    v,
                                    (
                                        "S16"
                                        if h5obj.dtype[v].kind == "O"
                                        else h5obj.dtype[v]
                                    ),
                                )
                                for v in h5obj.dtype.names
                            ]
                        elif self.vlen == "embed":
                            # embed fails due to https://github.com/zarr-developers/numcodecs/issues/333
                            data = h5obj[:].tolist()
                            data2 = []
                            for d in data:
                                data2.append(
                                    [
                                        (
                                            _.decode(errors="ignore")
                                            if isinstance(_, bytes)
                                            else _
                                        )
                                        for _ in d
                                    ]
                                )
                            dt = "O"
                            kwargs["data"] = data2
                            kwargs["object_codec"] = numcodecs.JSON()
                            fill = None
                        else:
                            raise NotImplementedError

                    if h5py.h5ds.is_scale(h5obj.id) and not cinfo:
                        return
                    if h5obj.attrs.get("_FillValue") is not None:
                        fill = encode_fill_value(
                            h5obj.attrs.get("_FillValue"), dt or h5obj.dtype
                        )

                # Create a Zarr array equivalent to this HDF5 dataset...
                za = self._zroot.create_dataset(
                    h5obj.name,
                    shape=h5obj.shape,
                    dtype=dt or h5obj.dtype,
                    chunks=h5obj.chunks or False,
                    fill_value=fill,
                    compression=None,
                    filters=filters,
                    overwrite=True,
                    **kwargs,
                )
                lggr.debug(f"Created Zarr array: {za}")
                self._transfer_attrs(h5obj, za)
                adims = self._get_array_dims(h5obj)
                za.attrs["_ARRAY_DIMENSIONS"] = adims
                lggr.debug(f"_ARRAY_DIMENSIONS = {adims}")

                if "data" in kwargs:
                    return  # embedded bytes, no chunks to copy

                # Store chunk location metadata...
                if cinfo:
                    for k, v in cinfo.items():
                        if h5obj.fletcher32:
                            logging.info("Discarding fletcher32 checksum")
                            v["size"] -= 4
                        if (
                            self.inline
                            and isinstance(v, dict)
                            and v["size"] < self.inline
                        ):
                            self.input_file.seek(v["offset"])
                            data = self.input_file.read(v["size"])
                            try:
                                # easiest way to test if data is ascii
                                data.decode("ascii")
                            except UnicodeDecodeError:
                                data = b"base64:" + base64.b64encode(data)
                            self.store[za._chunk_key(k)] = data
                        else:
                            self.store[za._chunk_key(k)] = [
                                self._uri,
                                v["offset"],
                                v["size"],
                            ]

            elif isinstance(h5obj, h5py.Group):
                lggr.debug(f"HDF5 group: {h5obj.name}")
                zgrp = self._zroot.create_group(h5obj.name)
                self._transfer_attrs(h5obj, zgrp)
        except Exception as e:
            import traceback

            msg = "\n".join(
                [
                    "The following excepion was caught and quashed while traversing HDF5",
                    str(e),
                    traceback.format_exc(limit=5),
                ]
            )
            if self.error == "ignore":
                return
            elif self.error == "pdb":
                print(msg)
                import pdb

                pdb.post_mortem()
            elif self.error == "raise":
                raise
            else:
                # "warn" or anything else, the default
                import warnings

                warnings.warn(msg)
            del e  # garbage collect
