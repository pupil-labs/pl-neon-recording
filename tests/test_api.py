import numpy as np

import pupil_labs.neon_recording as nr

# def test_package_metadata() -> None:
    # assert hasattr(nr, "__version__")


def test_nr_load_and_recarray_fields():
    rec = nr.load('./tests/test_data/2024-01-25_22-19-10_test-f96b6e36/')

    # gs = rec.streams['gaze']
    # ts = gaze.ts
    # gs.sample([ts[0], ts[1], ts[2]])

    # for g in gs:
    #     print(g)

    np.testing.assert_equal(rec.imu.data.tolist(), rec.streams['imu'].data.tolist())
    np.testing.assert_equal(rec.gaze.data.tolist(), rec.streams['gaze'].data.tolist())

    np.testing.assert_equal(rec.gaze.ts, rec.streams['gaze'].ts)
    np.testing.assert_equal(rec.imu.ts, rec.streams['imu'].ts)

    np.testing.assert_equal(rec.imu.ts_rel, rec.streams['imu'].ts_rel)
    np.testing.assert_equal(rec.gaze.ts_rel, rec.streams['gaze'].ts_rel)

