from dataclasses import dataclass
import os
from pathlib import Path
import traceback
from typing import Optional, Union
# Set tokenizer parallelism to false to avoid warnings when forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
from scipy.spatial.transform import Rotation as R

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig, CameraConfig

from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "PutRubbishInBin"         # Task suite
    # num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    # use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    # wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    # wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


class OpenVLAPolicy:
    """Adapter for OpenVLA model to conform to RLBench policy interface."""
    def __init__(self, cfg, action_head, proprio_projector, processor, vla, device):
        self.cfg = cfg
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.processor = processor
        self.vla = vla
        self.device = device
        print(f"OpenVLA policy initialized on device: {self.device}")

    def predict(self, observation, task_description):
        """Predict action given observation and text prompt"""
        
        cameras = ['left_shoulder_rgb', 'wrist_rgb']
        
        # Ensure cameras are uint8
        for camera_name in cameras:
            if not hasattr(observation, camera_name) or getattr(observation, camera_name) is None:
                raise ValueError(f"Observation missing required camera: {camera_name}")
            
            rgb_image = getattr(observation, camera_name)
            
            # Convert numpy array to PIL image
            if rgb_image.dtype != np.uint8:
                rgb_image = (rgb_image * 255).astype(np.uint8)
            # pil_image = Image.fromarray(rgb_image)
            setattr(self, camera_name, rgb_image)
        
        # Convert pose from quaternion to euler
        pose_euler = observation.gripper_pose # RLBench gives [xyz quat]
        if pose_euler.shape[-1] == 7:
            # convert quat to euler
            r = R.from_quat(pose_euler[3:7])  # (x, y, z, w)
            euler = r.as_euler('xyz', degrees=False)  # roll, pitch, yaw
            pose_euler = np.concatenate([pose_euler[0:3], euler])  # x, y, z, roll, pitch, yaw
        eef_pose = np.concatenate([pose_euler, np.array([observation.gripper_open], dtype=np.float32)])

        # Form prompt
        prompt = f"In: What action should the robot take to {task_description}?\nOut:"  # OpenVLA has this prompt structure!

        # Tokenize and process image
        inputs = {
            "full_image": observation.left_shoulder_rgb,
            "wrist_image": observation.wrist_rgb,
            "task_description": prompt,
            "joint_positions": observation.joint_positions.astype(np.float32),
            "eef_pose": eef_pose.astype(np.float32),
            "state": eef_pose.astype(np.float32),
        }
        
        # Predict action
        actions = get_vla_action(cfg=self.cfg,
                                 vla=self.vla,
                                 processor=self.processor,
                                 obs=inputs,
                                 task_label=prompt,
                                 action_head=self.action_head,
                                 proprio_projector=self.proprio_projector,
                                 use_film=True)
        
        # OpenVLA outputs: [x, y, z, rx, ry, rz, gripper]
        # RLBench expects: [x, y, z, qx, qy, qz, qw, gripper]

        for i, action in enumerate(actions):
            # Extract position, euler angles, and gripper
            pos = action[:3]
            euler = action[3:6]  # Euler angles in radians
            gripper = action[6:7]

            # Convert euler angles to quaternion
            rotation = Rotation.from_euler('xyz', euler)
            quat = rotation.as_quat()  # Returns [x, y, z, w]
            
            # Ensure quaternion is unit quaternion (normalize)
            # quat = quat / np.linalg.norm(quat)
            
            # Combine into RLBench action format: [x, y, z, qx, qy, qz, qw, gripper]
            actions[i] = np.concatenate([pos, quat, gripper])
        return actions


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
    
    print(f"Saving videos for episode {episode_idx} to {episode_dir} (n. frames per camera: {[len(frames) for frames in frames_dict.values()]})")
    
    for camera_name, frames in frames_dict.items():
        if frames:
            video_path = os.path.join(episode_dir, f"{camera_name}.mp4")
            # Ensure frames are uint8
            frames = [frame.astype(np.uint8) if frame.dtype != np.uint8 else frame for frame in frames]
            try:
                imageio.mimsave(video_path, frames, fps=fps)
            except Exception as e:
                print(f"Warning: Could not save video for {camera_name}: {e}")


def test_openvla(task_name, n_episodes, checkpoint_path):
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configuration
    results_dir = os.path.join("runs", "openvla_oft_test", f"{timestamp}_{task_name}")
    videos_dir = os.path.join(results_dir, "videos")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    
    print(f"Run directory: {results_dir}")
    print(f"Videos directory: {videos_dir}")
    print(f"Using checkpoint: {checkpoint_path}")

    # Load OpenVLA model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset stats used during finetuning (for action un-normalization)
    # dataset_statistics_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    # if os.path.isfile(dataset_statistics_path):
    #     print(f"Loading dataset statistics from {dataset_statistics_path}")
    #     with open(dataset_statistics_path, "r") as f:
    #         norm_stats = json.load(f)
    #     vla.norm_stats = norm_stats

    # Determine unnorm_key from checkpoint path
    unnorm_key = get_unnorm_key_from_checkpoint(checkpoint_path)
    
    # Instantiate config (see class GenerateConfig in experiments/robot/libero/run_libero_eval.py for definitions)
    cfg = GenerateConfig(
        pretrained_checkpoint = checkpoint_path,
        use_l1_regression = True,
        use_diffusion = False,
        use_film = True,
        num_images_in_input = 2,
        use_proprio = True,
        load_in_8bit = False,
        load_in_4bit = False,
        center_crop = True,
        num_open_loop_steps = NUM_ACTIONS_CHUNK,
        unnorm_key = unnorm_key,
    )
    
    # Load OpenVLA-OFT policy and inputs processor
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    
    # Load MLP action head to generate continuous actions (via L1 regression)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

    # Load proprio projector to map proprio to language embedding space
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=7) # PROPRIO_DIM)

    # Create policy
    policy = OpenVLAPolicy(cfg, action_head, proprio_projector, processor, vla, device)

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
        wrist_camera=camera_config,
        # front_camera=camera_config,
        joint_velocities=False,
        joint_positions=True,
        joint_forces=False,
        gripper_open=True,
        gripper_pose=True,
        task_low_dim_state=False
    )

    # Set up action mode (end-effector pose control)
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(
            absolute_mode=False,  # Use relative movements
            collision_checking=False,
        ),
        gripper_action_mode=Discrete()
    )

    # Create environment
    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=True
    )

    # Set max episode length
    max_steps = 300

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
            episode_frames = {cam: [getattr(obs, cam)] if hasattr(obs, cam) and getattr(obs, cam) is not None else [] for cam in camera_names}

            step_count = 0
            episode_reward = 0
            success = False
            
            # Create progress bar for this episode
            with tqdm(total=max_steps, desc=f"Episode {episode + 1} steps", unit="step") as pbar:
                
                while step_count < max_steps:
                    
                    # Get action from policy
                    try:
                        actions = policy.predict(obs, task_description)
                        
                        # Perform all actions in sequence
                        for action in actions:
                    
                            # Take step in environment
                            obs, reward, terminate = task_env.step(action)
                            episode_reward += reward
                            step_count += 1
                            
                            # Capture current frame for video
                            for cam_name in camera_names:
                                if hasattr(obs, cam_name) and getattr(obs, cam_name) is not None:
                                    frame = getattr(obs, cam_name)
                                    if frame.dtype != np.uint8:
                                        frame = (frame * 255).astype(np.uint8)
                                    episode_frames[cam_name].append(frame)

                            # Update progress bar
                            pbar.update(1)
                            pbar.set_postfix({'reward': f'{episode_reward:.3f}', 'step_reward': f'{reward:.3f}'})
                            
                            if terminate:
                                success = reward > 0.5  # Assume success if reward > 0.5
                                pbar.set_postfix({'reward': f'{episode_reward:.3f}', 'step_reward': f'{reward:.3f}', 'status': 'SUCCESS' if success else 'DONE'})
                                break
                            
                    except Exception as e:
                        print(f"Error during step {step_count} in episode {episode}: {e}")
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
    parser = argparse.ArgumentParser(description="Test OpenVLA-OFT policy on a specific RLBench task using direct RLBench interface")
    parser.add_argument("--task_name", type=str, choices=["put_rubbish_in_bin", "put_books_on_bookshelf", "empty_container"], help="Name of the RLBench task to test")
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of episodes to run for evaluation")
    parser.add_argument("--checkpoint", type=str, 
                       default="/storage/adelli/checkpoints/openvla_oft/openvla-7b+PutRubbishInBin_euler_relative+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--20000_chkpt",
                       help="Path to the OpenVLA model checkpoint directory")
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(project="openvla-oft-eval", 
               entity="andrea-delli-politecnico-di-torino", 
               name=f"openvla-oft")
    
    try:
        results = test_openvla(args.task_name, args.n_episodes, args.checkpoint)
        
        # Log results to wandb
        wandb.log({
            "success_rate": results[args.task_name]["success_rate"],
            "avg_reward": results[args.task_name]["avg_rewards"],
            "num_episodes": results[args.task_name]["num_episodes"]
        })
        
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        wandb.finish()
