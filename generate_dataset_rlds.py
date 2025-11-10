import envlogger
from envlogger.backends import tfds_backend_writer
import numpy as np
from tqdm import tqdm
import os
import tensorflow_datasets as tfds
import tensorflow as tf
import dm_env
from dm_env import specs
import json
import imageio
from scipy.spatial.transform import Rotation as R

from absl import app
from absl import flags

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
from rlbench.demo import Demo, ActionRepresentation
from play_demo import action_conversion, get_target_pose

# Actions are represented by end-effector position and rotation and gripper state. It can be absolute or relative to the current pose (delta).
# If action_dimension is 7, we have [x y z roll pitch yaw gripper_state], where rotation is in euler angles and gripper_state is 1 or 0.
# If action_dimension is 8, we have the rotation represented by a quaternion: [x y z qw qx qy qz gripper_state].

""" To get started:

export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

conda activate rlbench
python generate_dataset.py
"""

FLAGS = flags.FLAGS

def action_dimension():
    return 7 if FLAGS.action_repr == "euler" else 8

def action_repr() -> ActionRepresentation:
    return ActionRepresentation.EULER if FLAGS.action_repr == "euler" else ActionRepresentation.QUAT


def compute_and_save_norm_stats(dataset_path, out_name="norm_stats.json"):
    """
    Compute mean and std (per-element) for float observations and actions in a TFDS dataset
    located at `dataset_path` (builder directory). Saves results to dataset_path/out_name.

    Parameters:
        dataset_path: path to the TFDS builder directory
        out_name: output json filename (saved inside dataset_path)
    """
    builder = tfds.builder_from_directory(builder_dir=dataset_path)
    dataset = builder.as_dataset(split="train")

    norm_stats = {}  # key -> {"sum": np.array, "sum_sq": np.array, "count": int}

    for episode in tqdm(dataset, desc="Calculating normalization stats"):
        # episode is typically a mapping; ensure we iterate steps correctly
        # Some TFDS episode structures may have episode["steps"] as a tf.TensorArray-like
        steps = episode.get("steps", None)
        if steps is None:
            continue

        # iterate over steps (tf eager should make these Tensors iterable)
        for step in steps:
            # observations
            obs = step.get("observation", {})
            for key, value in obs.items():
                # Convert to numpy if Tensor, otherwise keep as is
                if hasattr(value, "numpy"):
                    value_np = value.numpy()
                else:
                    value_np = np.asarray(value)
                    
                if not hasattr(value_np, 'dtype'):
                    continue

                # only consider floating types
                if not np.issubdtype(value_np.dtype, np.floating):
                    continue

                # initialize accumulators as float64 arrays
                if key not in norm_stats:
                    norm_stats[key] = {
                        "sum": np.zeros_like(value_np, dtype=np.float64),
                        "sum_sq": np.zeros_like(value_np, dtype=np.float64),
                        "count": 0,
                    }

                # accumulate (cast to float64)
                val_f64 = value_np.astype(np.float64)
                norm_stats[key]["sum"] += val_f64
                norm_stats[key]["sum_sq"] += val_f64 * val_f64
                norm_stats[key]["count"] += 1

            # action (treat similarly)
            action = step.get("action", None)
            if action is not None:
                if hasattr(action, "numpy"):
                    action_np = action.numpy()
                else:
                    action_np = np.asarray(action)

                # only accumulate if numeric (float or int)
                if np.issubdtype(action_np.dtype, np.number):
                    key = "action"
                    if key not in norm_stats:
                        norm_stats[key] = {
                            "sum": np.zeros_like(action_np, dtype=np.float64),
                            "sum_sq": np.zeros_like(action_np, dtype=np.float64),
                            "count": 0,
                        }
                    a_f64 = action_np.astype(np.float64)
                    norm_stats[key]["sum"] += a_f64
                    norm_stats[key]["sum_sq"] += a_f64 * a_f64
                    norm_stats[key]["count"] += 1

    # finalize: compute mean and std, convert to lists for JSON
    result = {}
    for key, stats in norm_stats.items():
        cnt = int(stats["count"])
        if cnt == 0:
            # skip or set NaNs; here we'll set mean/std to None
            result[key] = {"mean": None, "std": None}
            continue

        mean = stats["sum"] / cnt
        mean_sq = stats["sum_sq"] / cnt
        var = mean_sq - mean * mean

        # numerical safety: clip small negative values to zero
        var = np.where(var < 0, 0.0, var)

        # choose ddof if sample std required (only if cnt > 1)
        if cnt > 1:
            # unbiased estimator: var * cnt / (cnt - 1)
            var = var * (cnt / (cnt - 1))

        std = np.sqrt(var)

        result[key] = { "mean": mean.tolist(), "std": std.tolist() }

    out_path = os.path.join(dataset_path, out_name)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Normalization stats saved to {out_path}")
    return result


def save_demo_video(demo, save_path, demo_idx=0, fps=10):
    """Save video files for a single demo using imageio."""
    if imageio is None:
        print("Skipping video saving: imageio not available. Install with 'pip install imageio'")
        return

    videos_path = os.path.join(save_path, "videos", FLAGS.task)
    os.makedirs(videos_path, exist_ok=True)

    demo_folder = os.path.join(videos_path, f"demo_{demo_idx:03d}")
    os.makedirs(demo_folder, exist_ok=True)

    camera_names = [
        "left_shoulder_rgb",
        # "right_shoulder_rgb",
        # "overhead_rgb",
        "wrist_rgb",
        "front_rgb",
    ]

    # print(f"Saving demo video for demo {demo_idx:03d}...")
    for camera_name in camera_names:
        frames = []
        for observation in demo:
            frame = getattr(observation, camera_name)
            if frame is not None:
                frames.append(frame)

        if frames:
            video_path = os.path.join(demo_folder, f"{camera_name}.mp4")
            frames = [
                frame.astype(np.uint8) if frame.dtype != np.uint8 else frame
                for frame in frames
            ]
            imageio.mimsave(video_path, frames, fps=fps)
    # print(f"Demo video saved to {demo_folder}")


# Insert class to replay recorded demos as episodes
class RLBenchDemoEnvWrapper(dm_env.Environment):
    """Wrapper that replays recorded demos as episodes for logging, loading one demo at a time."""

    def __init__(self, demos_dir):
        self.demos_dir = demos_dir
        self.demo_idx = -1
        self.step_idx = 0
        self.current_demo = None

        self.task_descriptions = {
            "PutRubbishInBin": "throw away the trash, leaving any other objects alone",
            "PutBooksOnBookshelf": "put 1 books on bookshelf",
            "EmptyContainer": "remove whatever you find in the big box in the middle and leave them in the red one"
        }

    def reset(self):
        self.demo_idx += 1
        self.step_idx = 0
        demo_file = os.path.join(self.demos_dir, f"demo_{self.demo_idx:03d}.pkl")
        self.current_demo = Demo.load(demo_file)
        obs = self.current_demo[self.step_idx]
        obs_dict = self._observation_to_dict(obs)
        return dm_env.restart(obs_dict)

    def step(self, action):
        self.step_idx += 1
        obs = self.current_demo[self.step_idx]
        obs_dict = self._observation_to_dict(obs)
        if self.step_idx == len(self.current_demo) - 1:
            return dm_env.termination(reward=0.0, observation=obs_dict)
        return dm_env.transition(reward=0.0, observation=obs_dict)

    def observation_spec(self):
        spec = {
            "left_shoulder_rgb": specs.Array(shape=(256, 256, 3), dtype=np.uint8, name="left_shoulder_rgb"),
            # "right_shoulder_rgb": specs.Array(shape=(256, 256, 3), dtype=np.uint8, name="right_shoulder_rgb"),
            # "overhead_rgb": specs.Array(shape=(256, 256, 3), dtype=np.uint8, name="overhead_rgb"),
            "wrist_rgb": specs.Array(shape=(256, 256, 3), dtype=np.uint8, name="wrist_rgb"),
            "front_rgb": specs.Array(shape=(256, 256, 3), dtype=np.uint8, name="front_rgb"),
            "joint_positions": specs.Array(shape=(7,), dtype=np.float32, name="joint_positions"),
            "eef_pose": specs.Array(shape=(7,), dtype=np.float32, name="eef_pose"), # [x y z roll pitch yaw gripper_open]
            "task_description": specs.StringArray(shape=(), name="task_description"),
        }
        return spec

    def action_spec(self):
        return specs.Array(shape=(action_dimension(),), dtype=np.float32, name="action")

    def reward_spec(self):
        return specs.Array(shape=(), dtype=np.float64, name="reward")

    def discount_spec(self):
        return specs.BoundedArray(shape=(), dtype=np.float64, minimum=0.0, maximum=1.0, name="discount")

    def _observation_to_dict(self, obs):
        
        # Convert from quaternion to euler
        pose_euler = obs.gripper_pose # RLBench gives [xyz quat]
        if pose_euler.shape[-1] == 7:
            # convert quat to euler
            r = R.from_quat(pose_euler[3:7])  # (x, y, z, w)
            euler = r.as_euler('xyz', degrees=False)  # roll, pitch, yaw
            pose_euler = np.concatenate([pose_euler[0:3], euler])  # x, y, z, roll, pitch, yaw

        eef_pose = np.concatenate([pose_euler, np.array([obs.gripper_open], dtype=np.float32)])
        
        # Get the correct task description, based on the current task name: one of the keys in self.task_descriptions should match the demos_dir
        task_name = os.path.basename(self.demos_dir)
        task_description = self.task_descriptions[task_name]

        return {
            "left_shoulder_rgb": obs.left_shoulder_rgb,
            # "right_shoulder_rgb": obs.right_shoulder_rgb,
            # "overhead_rgb": obs.overhead_rgb,
            "wrist_rgb": obs.wrist_rgb,
            "front_rgb": obs.front_rgb,
            "joint_positions": obs.joint_positions.astype(np.float32),
            "eef_pose": eef_pose.astype(np.float32),  # [x y z roll pitch yaw gripper_open]
            "task_description": task_description
        }


def main(argv):
    try:
        # Dynamically get the task class
        task_class = globals()[FLAGS.task]
    except KeyError:
        raise ValueError(f"Task {FLAGS.task} not found.")

    print(f"""Flags:
            save_path: "{FLAGS.save_path}",
            num_episodes: {FLAGS.num_episodes},
            task: {FLAGS.task},
            action_repr: {FLAGS.action_repr},
            absolute_actions: {FLAGS.absolute_actions},
            save_videos: {FLAGS.save_videos}""")

    # RLBench setup
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(absolute_mode=FLAGS.absolute_actions),
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
    # Generate RLDS dataset
    dataset_path = os.path.join(FLAGS.save_path,
                                "rlds",
                                f"{FLAGS.action_repr}_{'absolute' if FLAGS.absolute_actions else 'relative'}",
                                f"{FLAGS.task}_{FLAGS.action_repr}_{'absolute' if FLAGS.absolute_actions else 'relative'}",
                                "1.0.0")
    os.makedirs(dataset_path, exist_ok=True)

    # RLDS dataset configuration
    dataset_config = tfds.rlds.rlds_base.DatasetConfig(
        name=f"{FLAGS.task}_{FLAGS.action_repr}_{'absolute' if FLAGS.absolute_actions else 'relative'}",
        observation_info={
            "left_shoulder_rgb": tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8),
            # "right_shoulder_rgb": tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8),
            # "overhead_rgb": tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8),
            "wrist_rgb": tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8),
            "front_rgb": tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8),
            "joint_positions": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
            "eef_pose": tfds.features.Tensor(shape=(7,), dtype=tf.float32), # [x y z roll pitch yaw gripper_open]
            "task_description": tfds.features.Text(),
        },
        action_info=tfds.features.Tensor(shape=(action_dimension(),), dtype=tf.float64),
        reward_info=tf.float64,
        discount_info=tf.float64,
    )

    # Create the EnvLogger backend
    backend = tfds_backend_writer.TFDSBackendWriter(
        data_directory=dataset_path,
        split_name="train",
        max_episodes_per_file=100,
        ds_config=dataset_config,
    )

    # Wrap the canned demos as episodes for logging
    wrapped_env = RLBenchDemoEnvWrapper(demos_dir)
    with envlogger.EnvLogger(wrapped_env, backend=backend) as env_logger:
        for demo_idx in tqdm(range(FLAGS.num_episodes), desc="Recording demonstrations"):
            env_logger.reset()
            demo_file = os.path.join(demos_dir, f"demo_{demo_idx:03d}.pkl")
            demo = Demo.load(demo_file)

            current_pose = get_target_pose(demo, 0)
            for i, _ in enumerate(demo.actions):

                current_pose = get_target_pose(demo, i)
                next_pose = get_target_pose(demo, i + 1 if i + 1 < len(demo) else i)    # next pose (the one to reach)

                action = action_conversion(next_pose, FLAGS.action_repr, not FLAGS.absolute_actions, current_pose)
                timestep = env_logger.step(action)
                if timestep.last():
                    break

            if FLAGS.save_videos:
                save_demo_video(demo, FLAGS.save_path, demo_idx)

    print("Dataset generation complete.")
    env.shutdown()

    compute_and_save_norm_stats(dataset_path)


if __name__ == "__main__":
    flags.DEFINE_string("save_path", os.path.join(os.getcwd(), "datasets"), "Path to save the RLDS dataset.")
    flags.DEFINE_integer("num_episodes", 100, "Number of demonstrations to record.")
    flags.DEFINE_string("task", "PutRubbishInBin", "Name of the RLBench task.")
    flags.DEFINE_enum("action_repr", "quat", ["euler", "quat"], "Action representation.", required=False)
    flags.DEFINE_boolean("absolute_actions", True, "Whether to use absolute actions (True) or relative actions (False).", required=False)
    flags.DEFINE_boolean("save_videos", True, "Whether to save videos of the demonstrations.", required=False)
    app.run(main)
