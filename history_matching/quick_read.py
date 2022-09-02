import hashlib
import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)


def md5(fname):

    hash_md5 = hashlib.md5()

    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def quick_read_xl(excel_fn, sheet_name, force_read=False, **kwargs):

    excel_md5 = md5(excel_fn)

    filename = os.path.splitext(excel_fn)[0]
    hdf_fn = os.path.join("%s_%s.hd5" % (filename, excel_md5))

    if not force_read and os.path.isfile(hdf_fn):

        try:
            return quick_read_hdf(hdf_fn, sheet_name)
        except RuntimeError as rt:
            logger.error(rt)

    # Not in store, read and store now
    logger.info(f"Reading '{sheet_name}' from '{excel_fn}'...")
    sheet_data = pd.read_excel(excel_fn, sheet_name=sheet_name, **kwargs)

    store = pd.HDFStore(hdf_fn)
    store[sheet_name] = sheet_data
    store.close()

    return sheet_data


def quick_read_hdf(hdf_fn, sheet_name):

    logger.info(f"Reading '{sheet_name}' from '{hdf_fn}'...")
    store = pd.HDFStore(hdf_fn)

    if sheet_name in store:

        sheet_data = store[sheet_name]
        store.close()

        return sheet_data

    raise RuntimeError(f"Sheet '{sheet_name}' not found in HDF file '{hdf_fn}'.")
