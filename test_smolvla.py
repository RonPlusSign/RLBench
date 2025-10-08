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
from PIL import Image
import argparse
from tqdm import tqdm
import wandb
from scipy.spatial.transform import Rotation
from datetime import datetime

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper, JointPositionActionMode
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig, CameraConfig
import traceback
import sys

from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, SmolVLAConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import lerobot.policies.smolvla


class SmolPolicy:
    """Adapter for SmolVLA model to conform to RLBench policy interface."""
    def __init__(self, model, preprocessor, postprocessor, device):
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        print(f"SmolVLA policy initialized on device: {device}")

    def predict(self, observation, task_description):
        """Predict action given observation and text prompt"""
        
        cameras = ["left_shoulder_rgb", "right_shoulder_rgb", "wrist_rgb", "front_rgb", "overhead_rgb"]
        # cameras = ["front_rgb", "wrist_rgb"]
        
        # Get RGB image from the specified camera
        for camera in cameras:
            if not hasattr(observation, camera) or getattr(observation, camera) is None:
                raise ValueError(f"No {camera} image available in observation")

            img = getattr(observation, camera)
            # if img.dtype != np.uint8: # Convert numpy array to PIL image
            #     img = Image.fromarray((img * 255).astype(np.uint8))

            # Convert action values to [0, 1] range # TODO: DONE TO TEST HuggingFaceVLA/smolvla_libero
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            
            img = np.transpose(img, (2, 0, 1)) # Convert images from (H, W, C) to (C, H, W)
            img = torch.from_numpy(img).float().to(self.device) # Convert to torch tensor and move to device
            
            # Save image back for observation
            img_to_save = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            path = f"debug_{camera}.png"
            Image.fromarray(img_to_save).save(path)
            sys.exit(1)
            
            setattr(observation, camera, img)
            
        # Form prompt   # TODO: TEST ALSO WITHOUT "In: ...\nOut:"
        # prompt = f"In: What action should the robot take to {task_description}?\nOut:"
        prompt = task_description
        # print(f"  Joint positions: {observation.joint_positions}\n    Gripper_open: {observation.gripper_open}\n  Gripper_pose: {observation.gripper_pose}")  # DEBUG

        # Convert gripper_pose from [xyz qx qy qz qw] to [xyz rx ry rz rw]   # TODO: DONE TO TEST HuggingFaceVLA/smolvla_libero
        # gripper_pose = observation.gripper_pose  # [x, y, z, qx, qy, qz, qw]
        # rotation = Rotation.from_quat(gripper_pose[3:7])  # Convert quaternion to rotation object
        # euler = rotation.as_euler('xyz', degrees=False)  # Convert to Euler angles (rx, ry, rz)
        # gripper_pose_euler = np.concatenate([gripper_pose[:3], euler])  # [x, y, z, rx, ry, rz]

        data = {
            **{f"observation.images.{camera}": getattr(observation, camera) for camera in cameras},
            # "observation.images.image": getattr(observation, "front_rgb"),   # Use front camera
            # "observation.images.image2": getattr(observation, "wrist_rgb"),   # Use wrist camera
            
            "observation.state": torch.tensor(observation.gripper_pose.tolist() + [observation.gripper_open], dtype=torch.float32, device=self.device),
            # "observation.state": torch.tensor(gripper_pose_euler.tolist() + [observation.gripper_open], dtype=torch.float32, device=self.device),
            # "observation.state.joint": torch.tensor(observation.joint_positions.tolist(), dtype=torch.float32, device=self.device),
            "task_description": prompt,
            "task": prompt,
        }

        # Preprocess data
        data = self.preprocessor(data)

        # Predict action
        action_chunk = self.model.select_action(data)
        
        # Postprocess action
        action_chunk = self.postprocessor(action_chunk)
        
        # Convert actions from tensor to numpy
        actions = torch.tensor(action_chunk).cpu().numpy()
        
        # Add "1" as gripper open command (fully open)
        # actions = np.hstack([actions, np.ones((actions.shape[0], 1))])  # Gripper open command

        print(f"Predicted {actions.shape[0]} actions:\n{actions}")
        # return actions

        # SmolVLA outputs: [x, y, z, rx, ry, rz, gripper] for each action
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
        rlbench_actions = np.concatenate([pos, quat, gripper], axis=1)
        return rlbench_actions


def get_unnorm_key_from_checkpoint(checkpoint_path):
    """Determine the unnorm_key based on the checkpoint path."""
    checkpoint_name = os.path.basename(checkpoint_path)
    
    # Extract the pattern: <TaskName>_<euler/quat>_<relative/absolute>
    if '+' in checkpoint_name:
        # Extract everything after the '+' and before any additional '+'
        parts = checkpoint_name.split('+')
        if len(parts) >= 2:
            task_config = parts[1].split('+')[0]  # Get first part after '+', before any additional '+'
            unnorm_key = task_config
            print(f"Determined unnorm_key from checkpoint '{checkpoint_name}': {unnorm_key}")
            return unnorm_key
    
    # Fallback if pattern not found
    print(f"Warning: Could not parse checkpoint name '{checkpoint_name}', using default")
    return "PutRubbishInBin_euler_relative"


def save_episode_videos(frames_dict, save_dir, episode_idx, fps=10):
    """Save video files for each camera view of an episode."""
    episode_dir = os.path.join(save_dir, f"episode_{episode_idx:03d}")
    os.makedirs(episode_dir, exist_ok=True)
    
    for camera_name, frames in frames_dict.items():
        if frames:
            video_path = os.path.join(episode_dir, f"{camera_name}.mp4")
            # Ensure frames are uint8
            frames = [frame.astype(np.uint8) if frame.dtype != np.uint8 else frame for frame in frames]
            try:
                imageio.mimsave(video_path, frames, fps=fps)
            except Exception as e:
                print(f"Warning: Could not save video for {camera_name}: {e}")


def test_smolvla(task_name, n_episodes, checkpoint_path):
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configuration
    results_dir = os.path.join("runs", "smolvla_test", f"{timestamp}_{task_name}")
    videos_dir = os.path.join(results_dir, "videos")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    
    print(f"Run directory: {results_dir}")
    print(f"Videos directory: {videos_dir}")
    print(f"Using checkpoint: {checkpoint_path}")

    # Load SmolVLA model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Ensure the active CUDA device is GPU 0 if CUDA is available
    if device.type == "cuda":
        torch.cuda.set_device(0)
    print(f"Using device: {device}")

    # Create a trained policy
    config = make_policy_config("smolvla", max_state_dim=8, max_action_dim=8, resize_imgs_with_padding=(256, 256))
    # vla = make_policy(config, ds_meta=dataset.meta)
    vla = SmolVLAPolicy.from_pretrained(checkpoint_path)
    vla.to(device)
    
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=config)

    # Create policy
    policy = SmolPolicy(vla, preprocessor, postprocessor, device)

    # Set up RLBench environment
    # Configure cameras for observation and video recording
    camera_config = CameraConfig(
        rgb=True,
        depth=False,
        point_cloud=False,
        mask=False,
        image_size=(256, 256)
    )
    
    obs_config = ObservationConfig(
        left_shoulder_camera=camera_config,
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
    action_mode = JointPositionActionMode()
    # action_mode = MoveArmThenGripper(EndEffectorPoseViaPlanning(absolute_mode=True, collision_checking=False), Discrete())

    # Create environment
    env = Environment(action_mode=action_mode, obs_config=obs_config, headless=True)

    # Set max episode length
    # max_steps = 350 if task_name == "put_rubbish_in_bin" else (750 if task_name == "put_books_on_bookshelf" else 1000)
    max_steps = 150

    # Get task
    env.launch()
    task_class = env._string_to_task(task_name)
    task_env = env.get_task(task_class)

    print(f"Testing task: {task_name}")
    
    successes = []
    total_rewards = []
    
    camera_names = ['left_shoulder_rgb', 'overhead_rgb', 'wrist_rgb', 'front_rgb']
    
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
                            if frame.dtype != np.uint8:
                                frame = (frame * 255).astype(np.uint8)
                            episode_frames[cam_name].append(frame)
                    
                    # Get action from policy
                    try:
                        # if step_count < 10:
                        #     # For the first 10 steps, just go down by -0.05 on z-axis
                        #     # Create a simple downward action: [dx=0, dy=0, dz=-0.05, qx=0, qy=0, qz=0, qw=1, gripper=1]
                        #     action = np.array([0.0, 0.0, -0.01, 0.0, 0.0, 0.0, 1.0, 1.0])
                        #     print(f"Step {step_count + 1}: Using hardcoded downward action")
                        # else: # After 10 steps, use the normal policy
                        actions = policy.predict(obs, task_description)
                        
                        # The policy returns N actions, perform them all in sequence
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
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
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
    parser = argparse.ArgumentParser(description="Test SmolVLA policy on a specific RLBench task using direct RLBench interface")
    parser.add_argument("--task_name", type=str, 
                       choices=["put_rubbish_in_bin", "put_books_on_bookshelf", "empty_container"], 
                       help="Name of the RLBench task to test")
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of episodes to run for evaluation")
    parser.add_argument("--checkpoint", type=str, 
                    #    default="HuggingFaceVLA/smolvla_libero",
                       default="RonPlusSign/smolvla_PutRubbishInBin",
                       help="SmolVLA checkpoint to load from HuggingFace")
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(project="vla-smolvla-eval", entity="andrea-delli-politecnico-di-torino", name=f"smolvla-{args.task_name}")

    try:
        results = test_smolvla(args.task_name, args.n_episodes, args.checkpoint)

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
