import random

import numpy as np
import pupil_labs.neon_recording as nr

# TODO(rob) - turn this into an authentic test for pytest

rec = nr.load("./tests/test_data/2024-01-25_22-19-10_test-f96b6e36/")

gaze = rec.gaze
imus = rec.imu
eye = rec.eye
scene = rec.scene

gaze_test_tstamps = [
    gaze.ts[0],
    gaze.ts[1],
    gaze.ts[2],
    gaze.ts[len(gaze.ts) // 2],
    gaze.ts[-3],
    gaze.ts[-2],
    gaze.ts[-1],
]
shuffled_gaze_test_tstamps = random.sample(gaze_test_tstamps, len(gaze_test_tstamps))

imu_test_tstamps = [
    imus.ts[0],
    imus.ts[1],
    imus.ts[2],
    imus.ts[len(imus.ts) // 2],
    imus.ts[-3],
    imus.ts[-2],
    imus.ts[-1],
]
shuffled_imu_test_tstamps = random.sample(imu_test_tstamps, len(imu_test_tstamps))

gaze_samps = {"nearest_neighbor": [], "nearest_neighbor_shuffled": []}
imu_samps = gaze_samps.copy()

gaze_samps["nearest_neighbor"] = [s for s in gaze.sample(gaze_test_tstamps)]
gaze_samps["nearest_neighbor_shuffled"] = list(gaze.sample(shuffled_gaze_test_tstamps))

imu_samps["nearest_neighbor"] = [s for s in imus.sample(imu_test_tstamps)]
imu_samps["nearest_neighbor_shuffled"] = list(imus.sample(shuffled_imu_test_tstamps))

# basic test is that all of these methods should return the same result for each stream
all_equal = {"gaze": np.zeros((2, 2), dtype=bool), "imu": np.zeros((2, 2), dtype=bool)}
kc = 0
for k in gaze_samps:
    jc = 0
    for j in gaze_samps:
        all_equal["gaze"][kc, jc] = np.all(gaze_samps[k] == gaze_samps[j])
        all_equal["imu"][kc, jc] = np.all(imu_samps[k] == imu_samps[j])

        jc += 1

    kc += 1

print(all_equal)
print("all equal gaze:", np.all(all_equal["gaze"]))
print("all equal imu:", np.all(all_equal["imu"]))

# let's test linear interpolation methods
gaze_samps_linear = {
    "np.interp": [],
    "np.interp_shuffled": [],
    "nearest_neighbor": [],
    "nearest_neighbor_shuffled": [],
}

# np.interp approach
gaze_samps_linear["np.interp"] = list(gaze.sample(gaze_test_tstamps, method="linear"))
gaze_samps_linear["np.interp_shuffled"] = list(
    gaze.sample(shuffled_gaze_test_tstamps, method="linear")
)

# when asking linear interp to sample at tstamps that exist in the original data,
# we should get back the same results as nearest neighbor search on those
# same input tstamps

gaze_samps_linear["nearest_neighbor"] = list(gaze.sample(gaze_test_tstamps))
gaze_samps_linear["nearest_neighbor_shuffled"] = list(
    gaze.sample(shuffled_imu_test_tstamps)
)

# basic test is that all of these methods should return the same result for each stream
all_equal_interp = {
    "gaze": np.zeros((4, 4), dtype=bool),
}
kc = 0
for k in gaze_samps_linear:
    jc = 0
    for j in gaze_samps_linear:
        all_equal_interp["gaze"][kc, jc] = np.all(
            gaze_samps_linear[k] == gaze_samps_linear[j]
        )

        jc += 1

    kc += 1

print(all_equal_interp)
print("all equal gaze (linear interp):", np.all(all_equal_interp["gaze"]))
