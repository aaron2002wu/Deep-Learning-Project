import argparse
import csv
import os
import shutil
import subprocess
import pathlib
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

# ---------- Utility Functions ----------

def check_disk_usage(path="/"):
    """Check disk usage and warn if >90% used."""
    total, used, free = shutil.disk_usage(path)
    used_percent = used / total * 100
    print(f"[INFO] Disk usage: {used_percent:.2f}% used, {free // (1024**3)} GB free")
    if used_percent > 90:
        print("[WARNING] ⚠️ Storage critically low! Saving may fail.")
    return used_percent

def decompress_if_needed(path):
    """Automatically decompress .zstd bags."""
    p = pathlib.Path(path)
    if p.suffix == ".zstd":
        new_path = p.with_suffix("")  # remove .zstd
        print(f"[INFO] Decompressing {p} → {new_path}")
        subprocess.run(["unzstd", "-f", str(p)], check=True)
        return str(new_path)
    return str(p)

def flatten_msg(prefix, msg):
    """Recursively flatten a ROS2 message into dot-separated field paths with clean names."""
    items = {}
    if hasattr(msg, "__slots__"):
        for slot in msg.__slots__:
            value = getattr(msg, slot)
            slot_name = slot.lstrip('_')  # remove leading underscores
            full_key = f"{prefix}.{slot_name}" if prefix else slot_name

            # Recursively unpack nested messages
            if hasattr(value, "__slots__"):
                items.update(flatten_msg(full_key, value))
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    if hasattr(v, "__slots__"):
                        items.update(flatten_msg(f"{full_key}[{i}]", v))
                    else:
                        items[f"{full_key}[{i}]"] = v
            else:
                items[full_key] = value
    else:
        items[prefix] = msg
    return items


def read_messages(input_bag: str, target_topics):
    """Generator yielding (topic, msg, timestamp) tuples."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )

    topic_types = reader.get_all_topics_and_types()

    def typename(topic_name):
        for topic_type in topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"Topic {topic_name} not found in bag")

    print(f"[INFO] Reading messages from bag: {input_bag}")
    msg_count = 0
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic not in target_topics:
            continue
        msg_type = get_message(typename(topic))
        msg = deserialize_message(data, msg_type)
        msg_count += 1
        if msg_count % 500 == 0:
            print(f"[INFO] Processed {msg_count} messages...")
            check_disk_usage(os.path.expanduser("~/"))
        yield topic, msg, timestamp
    print(f"[INFO] Finished reading {msg_count} messages.")
    del reader

def clean_value(v):
    """Return a safe single-line string representation for CSV."""
    import numpy as np

    if isinstance(v, (list, tuple, np.ndarray)):
        # Flatten and convert to space-separated string
        flat = np.array(v).flatten()
        v_str = "[" + " ".join(map(str, flat)) + "]"
    else:
        v_str = str(v)
    # Remove any internal newlines or tabs
    return v_str.replace("\n", " ").replace("\t", " ").strip()

# ---------- Main Entry Point ----------

def main():
    parser = argparse.ArgumentParser(description="Convert ROS2 .mcap(.zstd) bag to flattened CSV")
    parser.add_argument(
        "--input",
        default=os.path.expanduser("~/Downloads/exp1/exp1_0.mcap"),
        help="Path to the input .mcap or .mcap.zstd file"
    )
    parser.add_argument(
        "--output",
        default=os.path.expanduser("~/Downloads/exp1/exp1_data.csv"),
        help="Output CSV file path"
    )
    args = parser.parse_args()

    # Auto-decompress if necessary
    args.input = decompress_if_needed(args.input)

    # Topics of interest
    target_topics = [
        "/bb04/experiment1/cmd_thrust",
        "/bb04/experiment1/mavros/global_position/global",
        "/bb04/experiment1/mavros/imu/data",
        "/bb04/experiment1/mavros/local_position/odom",
        "/bb04/experiment1/mavros/mission/waypoints"
    ]

    csv_file = args.output
    print(f"[INFO] Writing flattened data to: {csv_file}")

    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")  # tab-separated
        writer.writerow(["timestamp", "topic", "field", "value"])

        row_count = 0
        for topic, msg, timestamp in read_messages(args.input, target_topics):
            flat_fields = flatten_msg("", msg)
            for field_name, field_value in flat_fields.items():
                safe_value = clean_value(field_value)
                writer.writerow([int(timestamp / 1e9), topic, field_name, safe_value])
                row_count += 1


            if row_count % 1000 == 0:
                print(f"[INFO] Wrote {row_count} rows to CSV...")
                check_disk_usage(os.path.expanduser("~/"))

    print(f"[✅ DONE] Saved {row_count} total rows to {csv_file}")
    check_disk_usage(os.path.expanduser("~/"))

if __name__ == "__main__":
    main()

