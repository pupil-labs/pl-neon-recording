import sys

import numpy as np

from pupil_labs.neon_recording import NeonRecording


def main(rec_dir):
    rec = NeonRecording(rec_dir)

    # This is an example of how data streams can be accessed.
    # Gaze data will be used as the example, but the same applies to all streams.

    # All data streams feature timestamps. `ts` is an alias for `timestamps`.
    # Timestamps are absolute nanosecond integers since the unix epoch.
    timestamps = rec.gaze.abs_timestamp
    timestamps2 = rec.gaze.abs_ts
    assert np.all(timestamps == timestamps2)

    # Data streams can be indexed or sliced by their integer index
    g = rec.gaze[0]
    # Individual samples also feature timestamps
    print(g.abs_ts)
    g2 = rec.gaze[10:20]
    assert len(g2) == 10

    # Alternatively, data can be sliced by time as well
    ts = rec.gaze.abs_timestamp[0]
    g3 = rec.gaze.by_abs_timestamp[ts]
    assert g3 == g

    g = rec.gaze.by_abs_timestamp[ts : ts + 1e9]

    # In addtion to the absolute timestamps, relative timestamps are also available
    # These are timestamps in floating point seconds since the beginning of the
    # recording.
    ts = rec.gaze.rel_timestamp[0]
    rec.gaze.by_rel_timestamp[ts]
    rec.gaze.by_rel_timestamp[ts : ts + 1.0]


if __name__ == "__main__":
    main(sys.argv[1])
