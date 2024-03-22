import pprint
import sys

import pupil_labs.neon_recording as nr

rec = nr.load(sys.argv[1])

gaze = rec.gaze

print()
pprint.pprint(rec.unique_events)
print()

event1_ts = rec.unique_events["recording.begin"]
event2_ts = rec.unique_events["recording.end"]
between_two_events = gaze.ts[(gaze.ts >= event1_ts) & (gaze.ts <= event2_ts)]
print(event1_ts, event2_ts)

gaze_in_section = [s for s in gaze.sample(between_two_events)]
print("number of samples between the 2 events: " + str(len(gaze_in_section)))
print()
