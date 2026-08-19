import numpy as np

from pupil_labs.neon_recording.utils import load_multipart_data_time_pairs


def test_load_multipart_data_time_pairs_no_timestamps_dtype():
    empty_time_data_arrays = [(np.array([]), np.array([]))]
    dtype = np.dtype([("field1", "int64"), ("field2", "int64")])
    data = load_multipart_data_time_pairs(empty_time_data_arrays, dtype)

    assert list(data.dtype.fields) == ["time", "field1", "field2"]
