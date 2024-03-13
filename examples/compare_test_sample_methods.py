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

gaze_samps = {
    'searchsorted': [],
    'searchsorted_shuffled': [],
    'broadcast': [],
    'broadcast_shuffled': [],
    'rob_min': [],
    'rob_min_shuffled': []
}
imu_samps = gaze_samps.copy()

# nearest neighbor sampling
# np.searchsorted approach
gaze_samps['searchsorted'] = gaze.sample(test_tstamps)
gaze_samps['searchsorted_shuffled'] = gaze.sample(shuffled_test_tstamps)
imu_samps['searchsorted'] = imus.sample(test_tstamps)
imu_samps['searchsorted_shuffled'] = imus.sample(shuffled_test_tstamps)

# np broadcasting approach
gaze_samps['broadcast'] = gaze.sample_rob_broadcast(test_tstamps)
gaze_samps['broadcast_shuffled'] = gaze.sample_rob_broadcast(shuffled_test_tstamps)
imu_samps['broadcast'] = imus.sample_rob_broadcast(test_tstamps)
imu_samps['broadcast_shuffled'] = imus.sample_rob_broadcast(shuffled_test_tstamps)

# rob min diff approach
gaze_samps['rob_min'] = gaze.sample_rob(test_tstamps)
gaze_samps['rob_min_shuffled'] = gaze.sample_rob(shuffled_test_tstamps)
imu_samps['rob_min'] = imus.sample_rob(test_tstamps)
imu_samps['rob_min_shuffled'] = imus.sample_rob(shuffled_test_tstamps)

print()
print("imu methods are divergent because the test ts land between samples and nearest and insert order interp disagree")
print()

# basic test is that all of these methods should return the same result for each stream
all_equal = {
    'gaze': np.zeros((6, 6), dtype=bool),
    'imu': np.zeros((6, 6), dtype=bool)
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
    'rob.interp': [],
    'rob.interp_shuffled': [],
    'rob_min': [],
    'rob_min_shuffled': []
}

# np.interp approach
gaze_samps_linear['np.interp'] = gaze.sample(test_tstamps, method="linear")
gaze_samps_linear['np.interp_shuffled'] = gaze.sample(shuffled_test_tstamps, method="linear")

# rob approach
gaze_samps_linear['rob.interp'] = gaze.sample_rob_interp(test_tstamps)
gaze_samps_linear['rob.interp_shuffled'] = gaze.sample_rob_interp(shuffled_test_tstamps)

# when asking linear interp to sample at tstamps that exist in the original data,
# we should get back the same results as min diff, nearest neighbor search on those
# same input tstamps
gaze_samps_linear['rob_min'] = gaze.sample_rob(test_tstamps)
gaze_samps_linear['rob_min_shuffled'] = gaze.sample_rob(shuffled_test_tstamps)

# basic test is that all of these methods should return the same result for each stream
all_equal_interp = {
    'gaze': np.zeros((6, 6), dtype=bool),
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

# let's see if linearly interpolating between timestamps gives same results between np.interp
# and rob.interp. then, i feel fairly confident that all is working

test_tstamps = [tstamps[100], tstamps[150], tstamps[200], tstamps[len(tstamps)//2], tstamps[-400], tstamps[-203], tstamps[-200]]
for ic in range(len(test_tstamps)):
	test_tstamps[ic] = test_tstamps[ic] + np.random.uniform(0, 0.5)

shuffled_test_tstamps = random.sample(test_tstamps, len(test_tstamps))

gaze_samps_shifted = {
    'np.interp': [],
    'np.interp_shuffled': [],
    'rob.interp': [],
    'rob.interp_shuffled': [],
}

# np.interp approach
gaze_samps_shifted['np.interp'] = gaze.sample(test_tstamps, method="linear")
gaze_samps_shifted['np.interp_shuffled'] = gaze.sample(shuffled_test_tstamps, method="linear")

# rob approach
gaze_samps_shifted['rob.interp'] = gaze.sample_rob_interp(test_tstamps)
gaze_samps_shifted['rob.interp_shuffled'] = gaze.sample_rob_interp(shuffled_test_tstamps)

all_equal_interp_shifted = {
    'gaze': np.zeros((4, 4), dtype=bool),
}
kc = 0
for k in gaze_samps_shifted:
    jc = 0
    for j in gaze_samps_shifted:
        all_equal_interp_shifted['gaze'][kc, jc] =  np.all(gaze_samps_shifted[k] == gaze_samps_shifted[j])

        jc += 1

    kc += 1

print(all_equal_interp_shifted)
print('all equal gaze shifted (linear interp):', np.all(all_equal_interp_shifted['gaze']))
