import sys

import csv
import pupil_labs.neon_recording as nr

from tqdm import tqdm

def make_csv(recording_dir, csv_path):
    recording = nr.load(recording_dir)

    combined_data = zip(
        recording.imu,
        recording.gaze.sample(recording.imu.ts),
    )

    imu_fields = [f'imu_{f}' for f in recording.imu.data.dtype.names]
    gaze_fields = [f'gaze_{f}' for f in recording.gaze.data.dtype.names]

    fieldnames = imu_fields + gaze_fields
    with open(csv_path, 'wt') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for imu, gaze in tqdm(combined_data, total=len(recording.imu.ts)):
            imu_dict = dict(zip(imu_fields, imu))
            gaze_dict = dict(zip(gaze_fields, gaze))

            record = {**imu_dict, **gaze_dict}

            writer.writerow(record)

if __name__ == '__main__':
    make_csv(sys.argv[1], "matched-imu-gaze.csv")
