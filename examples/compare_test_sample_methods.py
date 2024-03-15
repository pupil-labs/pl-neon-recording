import random

import cv2
import numpy as np

import pupil_labs.neon_recording as nr

rec = nr.load('./tests/test_data/2024-01-25_22-19-10_test-f96b6e36/')

gaze = rec.gaze
imus = rec.imu
eye = rec.eye
scene = rec.scene

tstamps = gaze.ts
test_tstamps = [tstamps[0], tstamps[1], tstamps[2], tstamps[len(tstamps)//2], tstamps[-3], tstamps[-2], tstamps[-1]]
shuffled_test_tstamps = random.sample(test_tstamps, len(test_tstamps))

tstamps = imus.ts
imu_test_tstamps = [tstamps[0], tstamps[1], tstamps[2], tstamps[len(tstamps)//2], tstamps[-3], tstamps[-2], tstamps[-1]]
shuffled_imu_test_tstamps = random.sample(imu_test_tstamps, len(imu_test_tstamps))

gaze_samps = {
    'rob_min': [],
    'rob_min_shuffled': []
}
imu_samps = gaze_samps.copy()

gaze_samps['rob_min'] = [s for s in gaze.sample(test_tstamps)]
gaze_samps['rob_min_shuffled'] = [s for s in gaze.sample(shuffled_test_tstamps)]

imu_samps['rob_min'] = [s for s in imus.sample(test_tstamps)]
imu_samps['rob_min_shuffled'] = [s for s in imus.sample(shuffled_test_tstamps)]

# basic test is that all of these methods should return the same result for each stream
all_equal = {
    'gaze': np.zeros((2, 2), dtype=bool),
    'imu': np.zeros((2, 2), dtype=bool)
}
kc = 0
for k in gaze_samps:
    jc = 0
    for j in gaze_samps:
        all_equal['gaze'][kc, jc] =  np.all(gaze_samps[k] == gaze_samps[j])
        all_equal['imu'][kc, jc] =  np.all(imu_samps[k] == imu_samps[j])

        jc += 1

    kc += 1

print(all_equal)
print('all equal gaze:', np.all(all_equal['gaze']))
print('all equal imu:', np.all(all_equal['imu']))

# let's test linear interpolation methods
gaze_samps_linear = {
    'np.interp': [],
    'np.interp_shuffled': [],
    'rob_min': [],
    'rob_min_shuffled': []
}

# np.interp approach
gaze_samps_linear['np.interp'] = [s for s in gaze.sample(test_tstamps, method="linear")]
gaze_samps_linear['np.interp_shuffled'] = [s for s in gaze.sample(shuffled_test_tstamps, method="linear")]

# when asking linear interp to sample at tstamps that exist in the original data,
# we should get back the same results as min diff, nearest neighbor search on those
# same input tstamps
gaze_samps_linear['rob_min'] = [s for s in gaze.sample(test_tstamps)]
gaze_samps_linear['rob_min_shuffled'] = [s for s in gaze.sample(shuffled_test_tstamps)]

# basic test is that all of these methods should return the same result for each stream
all_equal_interp = {
    'gaze': np.zeros((4, 4), dtype=bool),
}
kc = 0
for k in gaze_samps_linear:
    jc = 0
    for j in gaze_samps_linear:
        all_equal_interp['gaze'][kc, jc] =  np.all(gaze_samps_linear[k] == gaze_samps_linear[j])

        jc += 1

    kc += 1

print(all_equal_interp)
print('all equal gaze (linear interp):', np.all(all_equal_interp['gaze']))
