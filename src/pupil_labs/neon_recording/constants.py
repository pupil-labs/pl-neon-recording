import numpy as np

TIMESTAMP_FIELD_NAME = "ts"
TIMESTAMP_DTYPE = np.dtype([(TIMESTAMP_FIELD_NAME, np.int64)])
AV_INDEX_FIELD_NAME = "idx"
AV_INDEX_DTYPE = np.dtype([(AV_INDEX_FIELD_NAME, np.int64)])

DTYPE_START_TIMESTAMP_FIELD_NAME = "start_timestamp_ns"
DTYPE_END_TIMESTAMP_FIELD_NAME = "end_timestamp_ns"
DTYPE_TIMESTAMP_FIELD_NAME = "timestamp_ns"
