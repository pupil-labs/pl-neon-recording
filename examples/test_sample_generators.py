import numpy as np

import pupil_labs.neon_recording as nr

rec = nr.load('./tests/test_data/2024-01-25_22-19-10_test-f96b6e36/')

gaze = rec.gaze
ts = gaze.ts

subsample = gaze.sample_rob([ts[0], ts[1], ts[2]])
samps = [s for s in subsample]
print(samps)
print()
print(len(samps))

tts = np.sort([ts[0], ts[1], ts[2]])
for c in range(3):
	print(gaze._ts_oob(tts[c]))
	print(np.argmin(np.abs(gaze.ts - tts[c])))