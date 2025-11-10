"""
This script fixes the problem of incompatibility with datasets > 4.0
See https://github.com/huggingface/lerobot/issues/1538 for details.

REMEMBER TO USE THIS WITH datasets=3.6 AND TO CHANGE root1 TO YOUR DATASET PATH.
REMEMBER TO DELETE HF CACHE BEFORE UPLOADING! (rm -rf ~/.cache/huggingface)
"""

import os
import pyarrow.parquet as pq
from datasets import Dataset, Features, Sequence, Value, Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root1 = "./datasets/lerobot/euler_relative/PutRubbishInBin_euler_relative"

# correct schema
features = Features({
    "observation.state": Sequence(Value("float32"), length=7),
    "observation.state.joints": Sequence(Value("float32"), length=7),
    "action": Sequence(Value("float32"), length=7),
    # "task_description": Value("string"),
    "observation.images.left_shoulder_rgb": Image(),
    # "observation.images.right_shoulder_rgb": Image(),
    # "observation.images.overhead_rgb": Image(),
    "observation.images.front_rgb": Image(),
    "observation.images.wrist_rgb": Image(),
    "timestamp": Value("float32"),
    "frame_index": Value("int64"),
    "episode_index": Value("int64"),
    "index": Value("int64"),
    "task_index": Value("int64"),
})

def fix_file(path: str):
    # read old
    table = pq.read_table(path)

    # strip metadata
    schema = table.schema.remove_metadata()
    table = table.cast(schema)

    # make HF Dataset and recast
    ds = Dataset(table).cast(features)

    # overwrite in place
    tmp_path = path + ".tmp"
    ds.to_parquet(tmp_path)
    os.replace(tmp_path, path)

    print(f"fixed {path}")

# walk through all shards
# for root, _, files in os.walk(f"{root1}/data"):
#     for fname in files:
#         if fname.endswith(".parquet"):
#             fix_file(os.path.join(root, fname))

# print(f"\n all parquet shards in {root1} have been rewritten with Sequence schema.")

# load fixed dataset and push to hub
dataset = LeRobotDataset("RonPlusSign/PutRubbishInBin_25_episodes", root1)

paths = dataset.get_episodes_file_paths()
print(f"Dataset has {len(paths)} episodes after fixing.")
for p in paths:
    print(f"- {p}")

dataset.finalize()
dataset.push_to_hub()

print(f"\033[92mDataset saved to {root1} and pushed to HuggingFace Hub.\033[0m")
