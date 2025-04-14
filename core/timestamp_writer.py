# timestamp_writer.py

import os
import datetime


def save_timestamps(video_path: str, timestamps: list[float], output_dir: str):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_timestamps.txt")

    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        for seconds in timestamps:
            timestamp = str(datetime.timedelta(seconds=int(seconds)))
            f.write(f"{timestamp}\n")

    print(f"[✔] Saved timestamps for {video_name} to {output_path}")
