import os
import numpy as np
import scipy.io as sio

RAW_DATA_DIR = r"C:\Users\kkl24\Downloads\pq7vb-osfstorage-Track#3 Imagined speech classification-archive\Training set"
filepath = os.path.join(RAW_DATA_DIR, "Data_Sample01.mat")

mat = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
print("KEYS:", [k for k in mat if not k.startswith('__')])

v = mat['epo_train']
print("TYPE:", type(v), "SHAPE:", getattr(v, 'shape', None), "DTYPE:", getattr(v, 'dtype', None))

if isinstance(v, np.ndarray) and v.dtype == object:
    inner = v.item() if v.size == 1 else v[0]
    print("INNER:", type(inner))
    print("FIELDS:", inner._fieldnames)