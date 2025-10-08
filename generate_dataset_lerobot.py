import os
import json
import pickle
import warnings
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import imageio
from tqdm import tqdm

from absl import app
from absl import flags
from huggingface_hub import Repository, create_repo
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# RLBench
from play_demo import get_target_joints
from rlbench import Environment
from rlbench.tasks import PutRubbishInBin, PutBooksOnBookshelf, EmptyContainer
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.demo import Demo

# Helpers
from generate_dataset_rlds import action_repr, get_target_pose, action_conversion, action_dimension

FLAGS = flags.FLAGS

# ------------------------
# Validator
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
    # Dynamically get the task class
    try:
        task_class = globals()[FLAGS.task]
        LOW_DIM_STATE_SIZE = 91 if task_class == PutRubbishInBin else 308 if task_class == PutBooksOnBookshelf else 70
    except KeyError:
        raise ValueError(f"Task {FLAGS.task} not found.")

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

    # Create LeRobot dataset
    print(f"Creating LeRobot dataset for task: {FLAGS.task}")
    
    dataset_root = os.path.join(
        FLAGS.save_path,
        "lerobot",
        f"{FLAGS.action_repr}_{'absolute' if FLAGS.absolute_actions else 'relative'}",
        f"{FLAGS.task}_{FLAGS.action_repr}_{'absolute' if FLAGS.absolute_actions else 'relative'}",
    )
    # Remove the dataset root if already exists
    if os.path.exists(dataset_root):
        print(f"Dataset root {dataset_root} already exists. Removing it.")
        shutil.rmtree(dataset_root)

    fps = 10
    camera_names = ["left_shoulder_rgb", "right_shoulder_rgb", "front_rgb", "wrist_rgb", "overhead_rgb"]
    # repo_name = f"RLBench-{FLAGS.task}-{FLAGS.action_repr}-{'absolute' if FLAGS.absolute_actions else 'relative'}"
    repo_name = f"RLBench-{FLAGS.task}-joint_positions"
    repo_id = f"RonPlusSign/{repo_name}"
    
    task_descriptions = {   
        "PutRubbishInBin": "throw away the trash, leaving any other objects alone",
        "PutBooksOnBookshelf": "put 1 books on bookshelf",
        "EmptyContainer": "remove whatever you find in the big box in the middle and leave them in the red one"
    }
    
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=dataset_root,
        robot_type="franka",
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),  # pos(3) + quat(4) + gripper(1)
                "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
                "description": "End-effector position (x,y,z), orientation (qx,qy,qz,qw) and gripper state (0.0 closed, 1.0 open).",
            },

            # TODO: IF USING DELTA EEF ACTIONS, UNCOMMENT THESE
            # "action": {
            #     "dtype": "float32",
            #     "shape": (7 if FLAGS.action_repr == "euler" else 8,),
            #     "names": ["x", "y", "z"] + (["roll", "pitch", "yaw"] if FLAGS.action_repr == "euler" else ["qx", "qy", "qz", "qw"]) + ["gripper"],
            #     "description": f"Delta action applied at each step, in {'Euler' if FLAGS.action_repr == 'euler' else 'Quaternion'} representation [xyz+rotation+gripper].",
            # },
            # "observation.joint_positions": {
            #     "dtype": "float32",
            #     "shape": (7,),
            #     "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"],
            #     "description": "Robot joint positions (absolute rotations).",
            # },
            # "observation.gripper_open": {
            #     "dtype": "float32",
            #     "shape": (1,),
            #     "names": ["gripper_open"],
            #     "description": "Gripper open state (0.0 closed, 1.0 open).",
            # },
            
            # TODO: IF USING ABSOLUTE JOINT POSITION ACTIONS, USE THESE
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"],
                "description": "Absolute joint position action applied at each step [joint_1 to joint_7 + gripper].",
            },
            
            "task_description": {
                "dtype": "string",
                "shape": (1,),
                "description": "A natural language description of the task.",
            },
            "next.done": {
                "dtype": "boolean",
                "shape": (1,),
                "description": "Indicates the end of an episode ; True for the last frame in each episode.",
            },
            **{f"observation.images.{cam}": {
                    "dtype": "video",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channels"],
                    "info": { "video.fps": fps, "video.height": 256, "video.width": 256, "video.channels": 3, "video.is_depth_map": False, "has_audio": False }
                } for cam in camera_names
            }
        },
    )

    for episode_index in tqdm(range(FLAGS.num_episodes), desc="Creating LeRobot dataset"):
        with open(os.path.join(demos_dir, f"demo_{episode_index:03d}.pkl"), "rb") as f:
            demo = pickle.load(f)
            
        for frame_index, observation in enumerate(demo):
            # Create the frame data, following the same structure as the features defined above
            frame_data = {
                "observation.state": get_target_pose(demo, frame_index).astype(np.float32),
                
                # TODO: IF USING DELTA EEF ACTIONS, UNCOMMENT THESE
                # "action": action_conversion(
                #         get_target_pose(demo, frame_index + 1 if frame_index < len(demo) - 1 else frame_index),
                #         action_repr().value,
                #         not FLAGS.absolute_actions,
                #         get_target_pose(demo, frame_index)
                # ).astype(np.float32),
                # "observation.joint_positions": observation.joint_positions.astype(np.float32),
                # "observation.gripper_open": np.array([observation.gripper_open], dtype=np.float32),
                
                # TODO: IF USING ABSOLUTE JOINT POSITION ACTIONS, USE THESE
                "action": get_target_joints(demo, frame_index + 1 if frame_index < len(demo) - 1 else frame_index).astype(np.float32),
                
                "next.done": frame_index == len(demo) - 1,
                "task": task.get_name(),
                "task_description": task_descriptions[FLAGS.task]
            }
            for cam in camera_names:
                frame_data[f"observation.images.{cam}"] = getattr(observation, cam)
            
            # Save the frame
            dataset.add_frame(frame_data)
        dataset.save_episode()
        
    dataset.push_to_hub()
    print(f"\033[92mDataset saved to {dataset_root} and pushed to HuggingFace Hub: {repo_id}\033[0m")
    
    validate_with_lerobot(
        dataset_root,
        expected_episodes=FLAGS.num_episodes,
        expected_action_dim=action_dimension(),
        expected_state_dim=8,  # pos(3) + quat(4) + gripper(1)
        camera_names=camera_names,
    )

if __name__ == "__main__":
    flags.DEFINE_string("save_path", os.path.join(os.getcwd(), "datasets"), "Path to save the LeRobot dataset.")
    flags.DEFINE_integer("num_episodes", 100, "Number of demonstrations to record.")
    flags.DEFINE_string("task", "PutRubbishInBin", "Name of the RLBench task.")
    flags.DEFINE_enum("action_repr", "euler", ["euler", "quat"], "Action representation.", required=False)
    flags.DEFINE_boolean("absolute_actions", True, "Whether to use absolute actions (True) or relative actions (False).", required=False)
    app.run(main)
