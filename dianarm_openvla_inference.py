from dataclasses import dataclass
import os
from pathlib import Path
import traceback
from typing import Optional, Union

import torch
import imageio
import numpy as np
import argparse
from datetime import datetime
from scipy.spatial.transform import Rotation as R

from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# Set tokenizer parallelism to false to avoid warnings when forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"
IMAGES_SHAPE = (300, 300, 3)  # Input image shape for OpenVLA-OFT (same used for training)

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
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 10                    # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "PutBlueBallInContainer"  # Task suite
    # num_steps_wait: int = 10                       # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 300                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    # use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    # wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    # wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

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


def save_episode_videos(frames_dict, save_dir, fps=10):
    """Save video files for each camera view of an episode."""
    
    episode_idx = len(os.listdir(save_dir)) # Count existing episodes to index this one
    episode_dir = os.path.join(save_dir, f"episode_{episode_idx:03d}")
    os.makedirs(episode_dir, exist_ok=True)
    
    print(f"Saving videos for episode {episode_idx} to {episode_dir}...")
    
    for camera_name, frames in frames_dict.items():
        if frames:
            video_path = os.path.join(episode_dir, f"{camera_name}.mp4")
            # Ensure frames are uint8
            frames = [frame.astype(np.uint8) if frame.dtype != np.uint8 else frame for frame in frames]
            try:
                imageio.mimsave(video_path, frames, fps=fps)
            except Exception as e:
                print(f"Warning: Could not save video for {camera_name}: {e}")


def test_openvla(checkpoint_path):
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configuration
    results_dir = os.path.join("runs", "realrobot_openvla_oft", f"{timestamp}")
    videos_dir = os.path.join(results_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    print(f"Run directory: {results_dir}")
    print(f"Videos directory: {videos_dir}")
    print(f"Using checkpoint: {checkpoint_path}")

    # Load OpenVLA model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine unnorm_key from checkpoint path
    unnorm_key = get_unnorm_key_from_checkpoint(checkpoint_path)
    
    # Instantiate config (see class GenerateConfig in experiments/robot/libero/run_libero_eval.py for definitions)
    cfg = GenerateConfig(
        pretrained_checkpoint = checkpoint_path,
        use_l1_regression = True,
        use_diffusion = False,
        use_film = True,
        num_images_in_input = 1,
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

    # Initialize video recording
    episode_frames = []

    step_count = 0
    task_description = "put the blue ball in the container"

    # Form prompt with the same structure used during training of OpenVLA-OFT
    prompt = f"In: What action should the robot take to {task_description}?\nOut:"  # OpenVLA has this prompt structure!

    while True:  # Control loop

        try:
            
            image = ...  # TODO: get latest image from camera as numpy array (H,W,3) RGB
            
            # crop image as needed (center crop to IMAGES_SHAPE), same used for training
            h, w, _ = image.shape
            target_h, target_w, _ = IMAGES_SHAPE
            start_h = (h - target_h) // 2 - 20  # slight vertical offset
            start_w = (w - target_w) // 2
            image_cropped = image[start_h:start_h+target_h, start_w:start_w+target_w, :]
            
            """
            Input for OpenVLA policy for DIANA arm:
            - 'feedback': (8,) float32 array of joint positions (feedback from jointAngles topic)
            - 'observation': (300, 300, 3) uint8 array of RGB image from wrist camera (ensure to be uint8 and RGB, swap channels if needed)
            - 'task_description': str describing the task to be performed by the robot (use `prompt` variable defined above)
            """
            # TODO: Prepare input dictionary with necessary keys (latest values of feedback and observation image)
            inputs = ...

            # Predict action
            actions = get_vla_action(cfg=cfg,
                                    vla=vla,
                                    processor=processor,
                                    obs=inputs,
                                    task_label=prompt,
                                    action_head=action_head,
                                    proprio_projector=proprio_projector,
                                    use_film=True)

            # Note: Actions have dim=7 [x y z roll pitch yaw gripper], but the control topic wants an additional "0" at the end
            actions = [np.append(action, 0) for action in actions]
            
            # Perform all actions in sequence
            for action in actions:
                step_count += 1
        
                # Take step in environment
                # TODO: Send action to topic
                
                # Capture current frame for video
                if ...: # TODO: check camera frame exists
                    frame = ... # TODO: get frame
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    episode_frames.append(frame)
                
        except Exception | KeyboardInterrupt as e:
            print(f"Error during step {step_count}: {e}")
            traceback.print_exc()
            break
    
    # Capture final frame
    if ...: # TODO: check camera frame exists
        frame = ... # TODO: get frame
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        episode_frames.append(frame)
    
    # Save episode video
    save_episode_videos(episode_frames, videos_dir, fps=30)
    print(f"Video saved to {videos_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OpenVLA-OFT policy on a real robotic arm task")
    parser.add_argument("--checkpoint", type=str, 
                       default="/storage/adelli/checkpoints/openvla_oft/my_openvla_oft_checkpoint",
                       help="Path to the OpenVLA-OFT model checkpoint directory")
    args = parser.parse_args()
    
    test_openvla(args.checkpoint)