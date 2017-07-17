import hashlib, os
import pandas as pd

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def quick_read(excel_fn, sheetname):
    excel_md5 = md5(excel_fn)

    filename = os.path.splitext(excel_fn)[0]
    hdf_fn = os.path.join( '%s_%s.hd5'%(filename, excel_md5) )
    if os.path.isfile(hdf_fn):
        print 'Reading %s from %s' % (sheetname, hdf_fn)
        store = pd.HDFStore(hdf_fn)
        if sheetname in store:
            sheetdata = store[sheetname]
            store.close()
            return sheetdata

    # Not in store, read and store now
    print 'Reading %s from %s' % (sheetname, excel_fn)
    sheetdata = pd.read_excel(excel_fn, sheetname=sheetname)

    store = pd.HDFStore(hdf_fn)
    store[sheetname] = sheetdata
    store.close()

    return sheetdata
