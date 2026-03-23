import pandas as pd
import yaml
import os
import math

# 1. Load the data
data_path = "Processed Data/touch_data_fixed.psv.txt"
df = pd.read_csv(data_path, sep="|")

# 2. Calculate Basic Statistics
stats = {
    "data_file": str(data_path),
    "dataset_overview": {
        "total_responses_rows": len(df),
        "unique_participants_count": int(df["ResponseID"].nunique()),
        "unique_videos_count": int(df["Touch No."].nunique(dropna=True)),
    },
    "missing_data_counts": df.isnull().sum().to_dict(),
}

# 3. Videos watched per participant
videos_per_participant = df.groupby("ResponseID")["Touch No."].count()
stats["participant_engagement"] = {
    "min_videos_watched": int(videos_per_participant.min()),
    "max_videos_watched": int(videos_per_participant.max()),
    "average_videos_watched": round(float(videos_per_participant.mean()), 3),
    "std_dev_videos_watched": round(float(videos_per_participant.std()), 3),
    "median_videos_watched": int(videos_per_participant.median()),
    "q1_videos_watched": int(videos_per_participant.quantile(0.25)),
    "q3_videos_watched": int(videos_per_participant.quantile(0.75)),
}

# 4. Video Parameters & Participant Counts per Video
video_stats = {}
grouped_videos = df.groupby("Touch No.")

for video_id, group in grouped_videos:
    # Get raw values first
    raw_contact = group["Contact"].iloc[0]
    raw_direction = group["Direction"].iloc[0]
    raw_speed = group["Speed (cm/s)"].iloc[0]
    raw_force = group["Force"].iloc[0]

    # Safely convert to types, handling NaN (Not a Number) values
    contact = str(raw_contact) if pd.notna(raw_contact) else "N/A"
    direction = str(raw_direction) if pd.notna(raw_direction) else "N/A"
    speed = int(raw_speed) if pd.notna(raw_speed) else "N/A"
    force = str(raw_force) if pd.notna(raw_force) else "N/A"

    # Format video_id as int to prevent float keys if some IDs are read as floats
    clean_video_id = int(video_id) if pd.notna(video_id) else "Unknown"

    video_stats[f"Video_{clean_video_id}"] = {
        "parameters": {
            "contact": contact,
            "direction": direction,
            "speed_cm_s": speed,
            "force": force,
        },
        "participant_response_count": len(group),
    }

stats["video_details"] = video_stats

# 5. List Videos per Parameter Value safely
# We drop NaN values before getting the unique lists to prevent errors
stats["videos_by_parameter"] = {
    "by_force": {
        "light": str(
            df[df["Force"] == "light"]["Touch No."]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
        "strong": str(
            df[df["Force"] == "strong"]["Touch No."]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
    },
    "by_contact": {
        "finger": str(
            df[df["Contact"] == "finger"]["Touch No."]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
        "hand": str(
            df[df["Contact"] == "hand"]["Touch No."]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
    },
    "by_direction": {
        "horizontal": str(
            df[df["Direction"] == "horizontal"]["Touch No."]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
        "vertical": str(
            df[df["Direction"] == "vertical"]["Touch No."]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
    },
    "by_speed": {
        "3_cm_s": str(
            df[df["Speed (cm/s)"] == 3]["Touch No."]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
        "9_cm_s": str(
            df[df["Speed (cm/s)"] == 9]["Touch No."]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
        "18_cm_s": str(
            df[df["Speed (cm/s)"] == 18]["Touch No."]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
    },
}

# 6. Save to YAML
output_path = "Metadata/experimental_setup.yaml"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as file:
    file.write("# Experimental Setup Description and Statistics\n")
    yaml.dump(stats, file, default_flow_style=False, sort_keys=False)

print(f"Experimental setup statistics successfully written to {output_path}")
