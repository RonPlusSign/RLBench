import os
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import imageio
from tqdm import tqdm

from absl import app
from absl import flags
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# RLBench
from rlbench import Environment
from rlbench.tasks import PutRubbishInBin, PutBooksOnBookshelf, EmptyContainer
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.demo import Demo

# Helpers
from generate_dataset import action_repr, get_target_pose, action_conversion, action_dimension

FLAGS = flags.FLAGS

# ------------------------
# Episode writer
# ------------------------


def write_episode_parquet_and_videos(demo: Demo, episode_idx: int, dataset_root: str, camera_names: list, fps: int, action_repr: str, absolute_actions: bool, task_index: int = 0):
    """ Write one RLBench demo to parquet + mp4 videos """

    root = Path(dataset_root)
    data_chunk_dir = root / "data" / "chunk-000"
    videos_chunk_dir = root / "videos" / "chunk-000"
    data_chunk_dir.mkdir(parents=True, exist_ok=True)
    videos_chunk_dir.mkdir(parents=True, exist_ok=True)

    for cam in camera_names:
        (videos_chunk_dir / f"observation.images.{cam}").mkdir(parents=True, exist_ok=True)

    num_frames = len(demo)
    rows = []
    cam_frames = {cam: [] for cam in camera_names}

    for t in range(num_frames):
        pose = get_target_pose(demo, t)  # expected pos+quat
        if len(pose) == 7:
            gripper = getattr(demo[t], "gripper_open", 1.0)
            state_vec = list(pose) + [float(gripper)]
        else:
            state_vec = list(pose)

        if t < num_frames - 1:
            next_pose = get_target_pose(demo, t + 1)
        else:
            next_pose = pose

        action = action_conversion(next_pose, action_repr, not absolute_actions, pose)
        action_list = list(np.array(action, dtype=float).tolist())

        timestamp = float(t) / float(fps)
        row = {
            "observation.state": state_vec,
            "action": action_list,
            "timestamp": float(timestamp),
            "episode_index": int(episode_idx),
            "frame_index": int(t),
            "index": int(episode_idx * 1_000_000 + t),
            "next.done": bool(t == num_frames - 1),
            "task_index": task_index
        }
        rows.append(row)

        for cam in camera_names:
            frame_img = getattr(demo[t], cam, None)
            if frame_img is None:
                if cam_frames[cam]:
                    fallback = cam_frames[cam][-1]
                else:
                    fallback = np.zeros((256, 256, 3), dtype=np.uint8)
                cam_frames[cam].append(fallback)
            else:
                cam_frames[cam].append(frame_img.astype(np.uint8) if frame_img.dtype != np.uint8 else frame_img)

    df = pd.DataFrame(rows)
    parquet_path = data_chunk_dir / f"episode_{episode_idx:06d}.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    for cam in camera_names:
        video_path = videos_chunk_dir / f"observation.images.{cam}" / f"episode_{episode_idx:06d}.mp4"
        imageio.mimsave(str(video_path), cam_frames[cam], fps=fps)

    return num_frames


# ------------------------
# Meta + tasks writers
# ------------------------


def write_meta_and_tasks(dataset_root: str, dataset_name: str, episode_stats: list, camera_names: list, task_list: list, fps: int):
    root = Path(dataset_root)
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_frames = sum(e["num_steps"] for e in episode_stats)

    task_descriptions = {
        "put_rubbish_in_bin": "throw away the trash, leaving any other objects alone",
        "put_books_on_bookshelf": "put 1 books on bookshelf",
        "empty_container": "remove whatever you find in the big box in the middle and leave them in the red one"
    }

    info = {
        "dataset_name": dataset_name,
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": len(episode_stats),
        "chunks_size": len(episode_stats),
        "total_frames": total_frames,
        "total_tasks": len(task_list),
        "total_videos": len(episode_stats) * len(camera_names),
        "fps": fps,
        "splits": {"train": f"0:{len(episode_stats)}"},  # all episodes in train split
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "camera_keys": camera_names,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [8],  # pos(3) + quat(4) + gripper(1)
                "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
                "description": "End-effector position (x,y,z), orientation (qx,qy,qz,qw) and gripper state (0.0 closed, 1.0 open).",
            },
            "action": {
                "dtype": "float32",
                "shape": [7 if action_repr() == "euler" else 8],
                "names": ["x", "y", "z"] + (["roll", "pitch", "yaw"] if action_repr() == "euler" else ["qx", "qy", "qz", "qw"]) + ["gripper"],
                "description": f"Action applied at each step, in {'Euler' if action_repr() == 'euler' else 'Quaternion'} representation + gripper (0.0 close, 1.0 open).",
            },
            **{f"observation.images.{cam}": {
                    "dtype": "video",
                    "shape": [256, 256, 3],
                    "names": ["height", "width", "channels"],
                    "info": { "video.fps": fps, "video.height": 256, "video.width": 256, "video.channels": 3, "video.is_depth_map": False, "has_audio": False }
                } for cam in camera_names
            }
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    with open(meta_dir / "episodes.jsonl", "w") as ef:
        for e in episode_stats:
            item = {
                "episode_index": int(e["episode_index"]),
                "tasks": [task_descriptions[task_list[0]["task"]]],
                "length": int(e["num_steps"]),
            }
            ef.write(json.dumps(item) + "\n")

    with open(meta_dir / "tasks.jsonl", "w") as tfp:
        for t in task_list:
            tfp.write(json.dumps({ "task_index": t["task_index"], "task": task_descriptions[t["task"]] }) + "\n")

    with open(meta_dir / "episodes_stats.jsonl", "w") as esf:
        for e in episode_stats:
            esf.write(json.dumps(e) + "\n")

    with open(root / "README.md", "w") as r:
        r.write(f"# {dataset_name}\n\nGenerated RLBench -> LeRobot v2.1 dataset.\n `meta/info.json`:\n```json\n{json.dumps(info, indent=4)}\n```\n")


# ------------------------
# Norm stats computation
# ------------------------


def compute_norm_stats_from_parquet(dataset_root: str, out_name="norm_stats.json"):
    root = Path(dataset_root)
    data_chunk = root / "data" / "chunk-000"
    parquet_files = sorted(data_chunk.glob("episode_*.parquet"))
    if not parquet_files:
        print("No parquet files found for norm stats.")
        return {}

    accum = {}
    for pqf in tqdm(parquet_files, desc="Computing norm stats"):
        df = pd.read_parquet(pqf)
        for col in ["action", "observation.state"]:
            if col not in df.columns:
                continue
            for arr in df[col].values:
                arr_np = np.asarray(arr, dtype=np.float64)
                if col not in accum:
                    accum[col] = {"sum": np.zeros_like(arr_np), "sum_sq": np.zeros_like(arr_np), "count": 0}
                accum[col]["sum"] += arr_np
                accum[col]["sum_sq"] += arr_np * arr_np
                accum[col]["count"] += 1

    result = {}
    for key, stats in accum.items():
        cnt = stats["count"]
        mean = stats["sum"] / cnt
        var = stats["sum_sq"] / cnt - mean * mean
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        result[key] = {"mean": mean.tolist(), "std": std.tolist()}

    out_path = root / "meta" / out_name
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Saved norm stats to {out_path}")
    return result


def compute_episode_stats(demo: Demo, camera_names: list = []):
    """ Compute statistics for one episode/demo (min, max, mean, std) for each feature.
    The analyzed features are:
    - observation.state
    - action
    - observation.images.{camera} (stats are per-channel)
    """

    states = np.array([get_target_pose(demo, i) for i in range(len(demo))])
    actions = np.array([action_conversion(get_target_pose(demo, i+1 if i < len(demo)-1 else i), action_repr().value, not FLAGS.absolute_actions, get_target_pose(demo, i)) for i in range(len(demo))])

    images = {cam: np.array([getattr(demo._observations[i], cam) for i in range(len(demo))]) for cam in camera_names}

    stats = {}
    if len(states) > 0:
        stats["observation.state"] = {
            "min": states.min(axis=0).tolist(),
            "max": states.max(axis=0).tolist(),
            "mean": states.mean(axis=0).tolist(),
            "std": states.std(axis=0).tolist(),
            "count": [len(states)]
        }

    if len(actions) > 0:
        stats["action"] = {
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
            "count": [len(actions)]
        }
        
    for cam, imgs in images.items():
        if len(imgs) > 0:
            # Compute stats per channel (reduce over frames, height, width)
            cam_stats = {
                "min": imgs.min(axis=(0, 1, 2)).reshape(3, 1, 1).tolist(),
                "max": imgs.max(axis=(0, 1, 2)).reshape(3, 1, 1).tolist(),
                "mean": imgs.mean(axis=(0, 1, 2)).reshape(3, 1, 1).tolist(),
                "std": imgs.std(axis=(0, 1, 2)).reshape(3, 1, 1).tolist(),
                "count": [len(imgs)]
            }
            stats[f"observation.images.{cam}"] = cam_stats
    
    return stats

# ------------------------
# Validator (warn-only)
# ------------------------


def validate_with_lerobot(dataset_root: str, expected_episodes: int, expected_action_dim: int, expected_state_dim: int, camera_names: list):
    dataset = LeRobotDataset(dataset_root)

    sample = dataset[0] # Get the first frame
    act = sample.get("action", None)
    st = sample.get("observation.state", None)

    if act is None or st is None:
        warnings.warn("Validator: missing action/state")
        return False

    if act.shape[-1] != expected_action_dim:
        warnings.warn(f"Validator: action dim {act.shape[-1]} != {expected_action_dim}")

    if st.shape[-1] != expected_state_dim:
        warnings.warn(f"Validator: state dim {st.shape[-1]} != {expected_state_dim}")

    for cam in camera_names:
        if f"observation.images.{cam}" not in sample:
            warnings.warn(f"Validator: missing camera {cam}")

    print("\033[92mValidator: loaded dataset successfully!\033[0m")
    return True


# ------------------------
# Main
# ------------------------


def main(argv):
    try:
        # Dynamically get the task class
        task_class = globals()[FLAGS.task]
        LOW_DIM_STATE_SIZE = 91 if task_class == PutRubbishInBin else 308 if task_class == PutBooksOnBookshelf else 70
    except KeyError:
        raise ValueError(f"Task {FLAGS.task} not found.")
    fps = 10

    # RLBench setup
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=FLAGS.absolute_actions),
        gripper_action_mode=Discrete(),
    )
    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()
    task = env.get_task(task_class)

    # Collect demonstrations if missing
    demos_dir = os.path.join(FLAGS.save_path, "demos", FLAGS.task)
    print(f"Generating {FLAGS.num_episodes} demos for task: {FLAGS.task}")
    os.makedirs(demos_dir, exist_ok=True)
    for i in tqdm(range(FLAGS.num_episodes), desc="Generating demos"):
        demo_file = os.path.join(demos_dir, f"demo_{i:03d}.pkl")
        if not os.path.exists(demo_file):
            # generate a new demo and save for future reuse
            demo = task.get_demos(1, live_demos=True)[0]
            demo.save(demo_file, action_representation=action_repr())
            del demo
    env.shutdown()

    # Convert demos -> parquet/videos
    dataset_root = os.path.join(
        FLAGS.save_path,
        "lerobot",
        f"{FLAGS.action_repr}_{'absolute' if FLAGS.absolute_actions else 'relative'}",
        f"{FLAGS.task}_{FLAGS.action_repr}_{'absolute' if FLAGS.absolute_actions else 'relative'}",
    )

    camera_names = ["left_shoulder_rgb", "right_shoulder_rgb", "front_rgb", "wrist_rgb", "overhead_rgb"]

    episode_stats = []
    for epi in tqdm(range(FLAGS.num_episodes), desc="Converting demos"):
        with open(os.path.join(demos_dir, f"demo_{epi:03d}.pkl"), "rb") as f:
            demo = pickle.load(f)
        n_steps = write_episode_parquet_and_videos(
            demo=demo,
            episode_idx=epi,
            dataset_root=dataset_root,
            camera_names=camera_names,
            fps=fps,
            action_repr=FLAGS.action_repr,
            absolute_actions=FLAGS.absolute_actions,
            task_index=0,
        )

        episode_stats.append({"episode_index": epi, "stats": compute_episode_stats(demo, camera_names), "num_steps": n_steps})

    task_list = [{"task_index": 0, "task": task.get_name()}]
    dataset_name = f"{task.get_name()}_{action_repr}_{'abs' if FLAGS.absolute_actions else 'rel'}"
    write_meta_and_tasks(dataset_root, dataset_name, episode_stats, camera_names, task_list, fps)

    compute_norm_stats_from_parquet(dataset_root)

    validate_with_lerobot(
        dataset_root,
        expected_episodes=FLAGS.num_episodes,
        expected_action_dim=action_dimension(),
        expected_state_dim=8,  # pos(3) + quat(4) + gripper(1)
        camera_names=camera_names,
    )

    print(f"\033[92mDataset saved to {dataset_root}\033[0m")
    
    return
    # Push the dataset to HuggingFace Hub if doesn't exist already
    try:
        from huggingface_hub import Repository, create_repo
        
        repo_name = f"rlbench-{FLAGS.task}-{FLAGS.action_repr}-{'absolute' if FLAGS.absolute_actions else 'relative'}"
        repo_id = f"RonPlusSign/{repo_name}"
        
        print(f"Pushing dataset to HuggingFace Hub: {repo_id}")
        create_repo(repo_id, exist_ok=True, private=False, repo_type="dataset")
        repo = Repository(local_dir=dataset_root, clone_from=repo_id, repo_type="dataset")
        repo.git_add(auto_lfs_track=True)
        repo.git_commit("Initial commit")
        repo.git_push()
        print(f"\033[92mDataset pushed to HuggingFace Hub: https://huggingface.co/datasets/{repo_id}\033[0m")
    except Exception as e:
        print(f"\033[93mWarning: could not push dataset to HuggingFace Hub: {e}\033[0m")


if __name__ == "__main__":
    flags.DEFINE_string("save_path", os.path.join(os.getcwd(), "datasets"), "Path to save the LeRobot dataset.")
    flags.DEFINE_integer("num_episodes", 100, "Number of demonstrations to record.")
    flags.DEFINE_string("task", "PutRubbishInBin", "Name of the RLBench task.")
    flags.DEFINE_enum("action_repr", "euler", ["euler", "quat"], "Action representation.", required=False)
    flags.DEFINE_boolean("absolute_actions", True, "Whether to use absolute actions (True) or relative actions (False).", required=False)
    app.run(main)
