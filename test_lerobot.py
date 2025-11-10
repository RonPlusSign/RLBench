import os

# Set tokenizer parallelism to false to avoid warnings when forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force usage of a single GPU to avoid mixed-device errors when models use device_map
# Set this BEFORE importing torch/transformers to ensure consistent device visibility
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
import torch
import imageio
import numpy as np
import argparse
from tqdm import tqdm
import wandb
from scipy.spatial.transform import Rotation
from datetime import datetime

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper, JointPositionActionMode
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, EndEffectorPoseViaPlanning, JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig, CameraConfig
import traceback
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy_config, make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.utils import build_inference_frame
from lerobot.datasets.utils import dataset_to_policy_features

OBS_CAMERA_KEYS = [ "left_shoulder_rgb", "right_shoulder_rgb", "front_rgb", "wrist_rgb", "overhead_rgb" ]
OBS_STATE_KEYS = [ "x", "y", "z", "roll", "pitch", "yaw", "gripper" ]
OBS_STATE_JOINT_KEYS = [ f"joint_{i+1}" for i in range(7) ]  # 7 joints for Franka Emika Panda
ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper_open"]
DEFAULT_IMAGE_SHAPE = (256, 256, 3)

class LeRobotPolicy:
    """Adapter for a LeRobot model to conform to RLBench policy interface."""
    def __init__(self, model, preprocessor, postprocessor, dataset_features, device):
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.dataset_features = dataset_features
        self.cameras = OBS_CAMERA_KEYS

        print(f"Policy initialized on device: {device}")

    def predict(self, observation, task_description):
        """Predict action given observation and text prompt"""
        
        raw_observation: dict[str, np.ndarray | float] = {}
        
        # Get RGB image from the specified camera
        for camera in self.cameras:
            if not hasattr(observation, camera) or getattr(observation, camera) is None:
                raise ValueError(f"No {camera} image available in observation")
            raw_observation[camera] = getattr(observation, camera)

        # Form prompt
        # prompt = f"In: What action should the robot take to {task_description}?\nOut:"
        prompt = task_description

        # Convert gripper_pose from [x y z qx qy qz qw] to [x y z roll pitch yaw gripper_open]
        gripper_pose = observation.gripper_pose  # [x, y, z, qx, qy, qz, qw]
        rotation = Rotation.from_quat(gripper_pose[3:7])  # Convert quaternion to rotation object
        euler = rotation.as_euler('xyz', degrees=False)  # Convert to Euler angles (roll, pitch, yaw)

        state = np.concatenate([ # Dim=7: [ x, y, z, roll, pitch, yaw, gripper_open]
            gripper_pose[:3],
            euler,
            np.array([observation.gripper_open], dtype=np.float32),
        ])
        for key, value in zip(OBS_STATE_KEYS, state):
            raw_observation[key] = np.array(value, dtype=np.float32)
        
        state_joints = np.array(observation.joint_positions, dtype=np.float32)
        for key, joint_val in zip(OBS_STATE_JOINT_KEYS, state_joints):
            raw_observation[key] = np.array(joint_val, dtype=np.float32)

        # Print key and shape
        # print(f"Raw observations:")
        # for k, v in raw_observation.items():
        #     print(f"  {k}: {v.shape if isinstance(v, np.ndarray) else type(v)}")
            
        raw_observation["task"] = prompt
        processed_observation = build_inference_frame(
            raw_observation,
            device=self.device,
            ds_features=self.dataset_features,
            task=prompt,
        )

        # print(f"Processed observations:")
        # for k, v in processed_observation.items():
        #     print(f"  {k}: {v.shape if isinstance(v, np.ndarray) else type(v)}")

        # Preprocess data
        processed_observation = self.preprocessor(processed_observation)

        # Predict action
        with torch.inference_mode():
            action_chunk = self.model.select_action(processed_observation)

        # Postprocess action
        action_chunk = self.postprocessor(action_chunk)
        
        # Convert actions from tensor to numpy
        actions = action_chunk.cpu().numpy()
        
        # Round the last dimension (gripper) to 0 or 1 to avoid fractional gripper commands
        actions[:, -1] = np.round(actions[:, -1])
        
        print(f"Predicted {actions.shape[0]} actions:\n{actions}")    # DEBUG
        return actions


def save_episode_videos(frames_dict, save_dir, episode_idx, fps=10):
    """Save video files for each camera view of an episode."""
    episode_dir = os.path.join(save_dir, f"episode_{episode_idx:03d}")
    os.makedirs(episode_dir, exist_ok=True)
    
    for camera_name, frames in frames_dict.items():
        if frames:
            video_path = os.path.join(episode_dir, f"{camera_name}.mp4")
            # Ensure frames are uint8
            frames = [np.array(frame).astype(np.uint8) for frame in frames]
            try:
                imageio.mimsave(video_path, frames, fps=fps)
            except Exception as e:
                print(f"Warning: Could not save video for {camera_name}: {e}")


def convert_euler_to_quat(actions):
    """
    Convert actions from Euler angles to quaternions.
    Assumes actions shape is (N, 7) with [x, y, z, rx, ry, rz, gripper].
    Returns actions in shape (N, 8) with [x, y, z, qx, qy, qz, qw, gripper].
    """
    
    if actions.shape[1] != 7:
        raise ValueError(f"Expected actions shape (N, 7), got {actions.shape}")

    # Model outputs: [x, y, z, rx, ry, rz, gripper] for each action
    # RLBench expects: [x, y, z, qx, qy, qz, qw, gripper]
    
    # Extract position, euler angles, and gripper
    pos = actions[:, :3]
    euler = actions[:, 3:6]  # Euler angles in radians
    gripper = actions[:, 6:7]

    # Convert euler angles to quaternion
    rotation = Rotation.from_euler('xyz', euler)
    quat = rotation.as_quat()  # Returns [x, y, z, w]
    
    # Ensure quaternion is unit quaternion (normalize)
    quat = quat / np.linalg.norm(quat)
    
    # Combine into RLBench action format: [x, y, z, qx, qy, qz, qw, gripper]
    return np.concatenate([pos, quat, gripper], axis=1)


def test_model(task_name, n_episodes, checkpoint_path):
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_type = ""
    if "smolvla" in checkpoint_path.lower():
        model_type = "smolvla"
    elif "diffusion" in checkpoint_path.lower():
        model_type = "diffusion"
    elif "act" in checkpoint_path.lower():
        model_type = "act"
    else:
        raise ValueError("Model type not recognized from checkpoint path.")
    
    # Configuration
    results_dir = os.path.join("runs", f"{model_type}_test", f"{timestamp}_{task_name}")
    videos_dir = os.path.join(results_dir, "videos")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    
    print(f"Run directory: {results_dir}")
    print(f"Videos directory: {videos_dir}")
    print(f"Using checkpoint: {checkpoint_path}")

    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Ensure the active CUDA device is GPU 0 if CUDA is available
    if device.type == "cuda":
        torch.cuda.set_device(0)
    print(f"Using device: {device}")
    
    dataset_id = "RonPlusSign/PutRubbishInBin_RLBENCH"
    dataset_metadata = LeRobotDatasetMetadata(dataset_id) # This only downloads the metadata for the dataset
    features = dataset_to_policy_features(dataset_metadata.features)
    input_features = {k: v for k, v in features.items() if k != "action"}   # Exclude action from input features
    output_features = {"action": features["action"]}  # Only action is output feature
    
    print(f"Features:")
    for feature in features:
        print(f"  {feature}: {features[feature]}")

    # Create a trained policy
    args = {
        "n_action_steps": 1,
        "n_obs_steps": 1,
        # "horizon": 8,
        "input_features": input_features,
        "output_features": output_features,
    }
    
    config = make_policy_config(model_type, **args)
    
    if model_type == "act":
        vla = ACTPolicy.from_pretrained(checkpoint_path, config=config)
    elif model_type == "diffusion":
        vla = DiffusionPolicy.from_pretrained(checkpoint_path, config=config)
    elif model_type == "smolvla":
        vla = SmolVLAPolicy.from_pretrained(checkpoint_path, config=config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    vla.to(device)
    vla.eval()
        

    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=config, pretrained_path=checkpoint_path, dataset_stats=dataset_metadata.stats)
    
    # Create policy
    policy = LeRobotPolicy(vla, preprocessor, postprocessor, dataset_features=dataset_metadata.features, device=device)

    # Set up RLBench environment
    # Configure cameras for observation and video recording
    camera_config = CameraConfig(rgb=True, depth=False, point_cloud=False, mask=False, image_size=(256, 256))
    
    obs_config = ObservationConfig(
        left_shoulder_camera=camera_config,
        right_shoulder_camera=camera_config,
        overhead_camera=camera_config,
        wrist_camera=camera_config,
        front_camera=camera_config,
        joint_velocities=False,
        joint_positions=True,
        joint_forces=False,
        gripper_open=True,
        gripper_pose=True,
        task_low_dim_state=False
    )

    # Set up action mode (end-effector pose control)
    # action_mode = JointPositionActionMode()
    action_mode = MoveArmThenGripper(EndEffectorPoseViaIK(absolute_mode=False, collision_checking=False), Discrete())

    # Create environment
    env = Environment(action_mode=action_mode, obs_config=obs_config, headless=True)
    max_steps = 200 # Set max episode length

    # Get task
    env.launch()
    task_class = env._string_to_task(task_name)
    task_env = env.get_task(task_class)
    print(f"Testing task: {task_name}")
    
    successes = []
    total_rewards = []
    
    camera_names = ['left_shoulder_rgb', 'overhead_rgb', 'wrist_rgb', 'front_rgb']  # To save episode videos
    
    for episode in range(n_episodes):
        try:
            print(f"\n=== Episode {episode + 1}/{n_episodes} ===")
            
            # Reset environment
            descriptions, obs = task_env.reset()
            task_description = descriptions[0] if descriptions else task_name.replace('_', ' ')
            
            # Initialize video recording
            episode_frames = {cam: [] for cam in camera_names}
            
            step_count = 0
            episode_reward = 0
            success = False
            
            # Create progress bar for this episode
            with tqdm(total=max_steps, desc=f"Episode {episode + 1} steps", unit="step") as pbar:
                while step_count < max_steps:
                    # Capture current frame for video
                    for cam_name in camera_names:
                        if hasattr(obs, cam_name) and getattr(obs, cam_name) is not None:
                            frame = getattr(obs, cam_name)
                            episode_frames[cam_name].append(frame)
                    
                    # Get action from policy
                    try:
                        actions = policy.predict(obs, task_description)
                        actions = convert_euler_to_quat(actions)  # Convert Euler angles to quaternions if needed
                        
                        # The policy may return N actions, perform them all in sequence
                        for act in actions:
                            # Take step in environment
                            obs, reward, terminate = task_env.step(act)
                            episode_reward += reward
                            step_count += 1

                            # Update progress bar
                            pbar.update(1)
                            pbar.set_postfix({'reward': f'{episode_reward:.3f}', 'step_reward': f'{reward:.3f}'})
                            
                            if terminate:
                                success = reward > 0.5  # Assume success if reward > 0.5
                                pbar.set_postfix({'reward': f'{episode_reward:.3f}', 'step_reward': f'{reward:.3f}', 'status': 'SUCCESS' if success else 'DONE'})
                                break
                            
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = exc_tb.tb_frame.f_code.co_filename
                        lineno = exc_tb.tb_lineno
                        print(f"[{fname}:{lineno}] Error during step {step_count} in episode {episode}: {e}")
                        traceback.print_exc()
                        pbar.set_postfix({'reward': f'{episode_reward:.3f}', 'status': 'ERROR'})
                        break
                    
            # Capture final frame
            for cam_name in camera_names:
                if hasattr(obs, cam_name) and getattr(obs, cam_name) is not None:
                    frame = getattr(obs, cam_name)
                    episode_frames[cam_name].append(frame)
            
            # Save episode video
            save_episode_videos(episode_frames, videos_dir, episode, fps=30)
            
            successes.append(success)
            total_rewards.append(episode_reward)
            
            print(f"Episode {episode + 1}: Steps={step_count}, Reward={episode_reward:.3f}, Success={success}")
            
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            successes.append(False)
            total_rewards.append(0)

    # Calculate metrics
    success_rate = np.mean(successes)
    avg_reward = np.mean(total_rewards)
    
    results = {
        task_name: {
            "success_rate": float(success_rate),
            "avg_rewards": float(avg_reward),
            "num_episodes": n_episodes,
            "successes": successes,
            "rewards": total_rewards
        }
    }

    print(f"Task {task_name}: success_rate={success_rate:.3f}, avg_rewards={avg_reward:.3f}")

    # Save results
    results_file = os.path.join(results_dir, f"results_{task_name}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {results_file}")
    print(f"Videos saved to {videos_dir}")

    # Shutdown environment
    env.shutdown()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a LeRobot policy on a specific RLBench task using direct RLBench interface")
    parser.add_argument("--task_name", type=str, choices=["put_rubbish_in_bin", "put_books_on_bookshelf", "empty_container"], help="Name of the RLBench task to test")
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of episodes to run for evaluation")
    parser.add_argument("--checkpoint", type=str, default="RonPlusSign/smolvla_PutRubbishInBin", help="Checkpoint to load from HuggingFace")
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(project="vla-smolvla-eval", entity="andrea-delli-politecnico-di-torino", name=f"lerobot-{args.task_name}")

    try:
        results = test_model(args.task_name, args.n_episodes, args.checkpoint)

        # Log results to wandb
        wandb.log({
            "success_rate": results[args.task_name]["success_rate"],
            "avg_reward": results[args.task_name]["avg_rewards"],
            "num_episodes": results[args.task_name]["num_episodes"]
        })
        
    except Exception as e:
        traceback.print_exc()
    finally:
        wandb.finish()
