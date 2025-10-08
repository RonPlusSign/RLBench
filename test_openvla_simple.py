import os
# Set tokenizer parallelism to false to avoid warnings when forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import imageio
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import argparse
from tqdm import tqdm
import wandb
from scipy.spatial.transform import Rotation
from datetime import datetime

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig, CameraConfig


class OpenVLAPolicy:
    """Adapter for OpenVLA model to conform to RLBench policy interface."""
    def __init__(self, processor, model, device, unnorm_key):
        self.processor = processor
        self.model = model
        self.device = device
        self.unnorm_key = unnorm_key
        print(f"OpenVLA policy initialized with unnorm_key: {self.unnorm_key}")

    def predict(self, observation, task_description):
        """Predict action given observation and text prompt"""
        
        # Get RGB image from the specified camera (left_shoulder by default)
        if hasattr(observation, 'left_shoulder_rgb') and observation.left_shoulder_rgb is not None:
            rgb_image = observation.left_shoulder_rgb
        else:
            raise ValueError("No RGB image available in observation")
        
        # Convert numpy array to PIL image
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        pil_image = Image.fromarray(rgb_image)

        # Form prompt
        # prompt = f"In: What action should the robot take to {task_description}?\nOut:"
        prompt = task_description

        # Tokenize and process image
        inputs = self.processor([prompt], [pil_image], return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(self.device) 
                 for k, v in inputs.items()}

        # Predict action
        action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        
        # Convert action from tensor to numpy
        action = torch.tensor(action).cpu().numpy()
        
        # OpenVLA outputs: [x, y, z, rx, ry, rz, gripper]
        # RLBench expects: [x, y, z, qx, qy, qz, qw, gripper]
        
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
        rlbench_action = np.concatenate([pos, quat, gripper])
        
        return rlbench_action


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


def test_openvla(task_name, n_episodes, checkpoint_path):
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configuration
    results_dir = os.path.join("runs", "openvla_simple_test", f"{timestamp}_{task_name}")
    videos_dir = os.path.join(results_dir, "videos")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    
    print(f"Run directory: {results_dir}")
    print(f"Videos directory: {videos_dir}")
    print(f"Using checkpoint: {checkpoint_path}")

    # Load OpenVLA model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)

    # Load dataset stats used during finetuning (for action un-normalization)
    dataset_statistics_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        print(f"Loading dataset statistics from {dataset_statistics_path}")
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats

    # Determine unnorm_key from checkpoint path
    unnorm_key = get_unnorm_key_from_checkpoint(checkpoint_path)

    # Create policy
    policy = OpenVLAPolicy(processor, vla, device, unnorm_key)

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
    max_steps = 350 if task_name == "put_rubbish_in_bin" else (750 if task_name == "put_books_on_bookshelf" else 1000)

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
                        # else: # After 10 steps, use the normal OpenVLA policy
                        action = policy.predict(obs, task_description)
                        
                        # Take step in environment
                        obs, reward, terminate = task_env.step(action)
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
                        print(f"Error during step {step_count} in episode {episode}: {e}")
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
    parser = argparse.ArgumentParser(description="Test OpenVLA policy on a specific RLBench task using direct RLBench interface")
    parser.add_argument("--task_name", type=str, 
                       choices=["put_rubbish_in_bin", "put_books_on_bookshelf", "empty_container"], 
                       help="Name of the RLBench task to test")
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of episodes to run for evaluation")
    parser.add_argument("--checkpoint", type=str, 
                       default="/home/adelli/openvla/checkpoints/openvla-7b+PutRubbishInBin_euler_relative+b10+lr-0.0005+lora-r32+dropout-0.0--image_aug",
                       help="Path to the OpenVLA model checkpoint directory")
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(project="vla-eval", 
               entity="andrea-delli-politecnico-di-torino", 
               name=f"openvla-simple-{args.task_name}")
    
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
