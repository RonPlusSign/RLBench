import os
import json
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PutRubbishInBin, PutBooksOnBookshelf, EmptyContainer

from LIFT3D.lift3d.envs.rlbench_env import RLBenchActionMode, RLBenchObservationConfig, RLBenchEnv

# Import openpi components
from openpi.training import config
from openpi.shared import download
from openpi.policies import policy_config

"""
RUN WITH:
export LIBGL_ALWAYS_SOFTWARE=1 && python test_openpi.py
"""

def test_openpi():
    # Configuration
    tasks = [PutRubbishInBin, PutBooksOnBookshelf, EmptyContainer]
    # Use the base pi0 model as specified in the openpi README
    model_name = "pi0_libero" # Corrected model name
    results_dir = "runs/openpi_test"
    videos_dir = os.path.join(results_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # Load OpenPI model
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # Assuming cuda:0 is available, adjust if needed
    
    # Get config and download checkpoint
    openpi_config = config.get_config(model_name)
    checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{model_name}")
    
    # Create a trained policy
    policy = policy_config.create_trained_policy(openpi_config, checkpoint_dir)

    # Setup RLBench environment
    obs_config = RLBenchObservationConfig.multi_view_config(
        camera_names=["front", "wrist", "left_shoulder", "right_shoulder", "overhead"],
        image_size=(256, 256)
    )

    action_mode = RLBenchActionMode.eepose_then_gripper_action_mode(absolute=False)

    env = RLBenchEnv(
        task_name="put_rubbish_in_bin", # Will be changed in the loop
        action_mode=action_mode,
        obs_config=obs_config,
        headless=True,
        cinematic_record_enabled=False, # Set to True if you want to record videos
        verbose_warnings=True,
    )
    
    results = {}
    
    for task_class in tasks:
        task_name = task_class.__name__
        task_name = ''.join(['_' + i.lower() if i.isupper() else i for i in task_name]).lstrip('_')
        env.task_name = task_name
        env.env.task_class = task_class
        print(f"Testing task: {task_name}")
        
        n_episodes = 10
        max_steps = 100
        success_rate = 0
        
        for i in tqdm(range(n_episodes), desc=f"Running {task_name} episodes"):
            obs = env.reset()
            
            # Get language instruction
            instruction = env.text
            
            for step in range(max_steps):
                # Prepare observation for OpenPI
                # OpenPI expects observations in a specific dictionary format
                # Mapping RLBench observations to OpenPI's expected format
                openpi_obs = {
                    "prompt": instruction,
                    "observation/image": Image.fromarray(obs['image']),
                    "observation/state": obs['robot_state'],
                }
                
                # Run inference
                action_chunk = policy.infer(openpi_obs)["actions"]
                
                # OpenPI returns action_chunk, which might need processing to fit RLBench's step
                # Assuming action_chunk is a numpy array or similar that can be directly used
                # You might need to convert/reshape action_chunk based on RLBench's action_mode
                action = action_chunk[0] # Take the first action from the chunk if it's a sequence
                
                print(f"Step {step + 1}/{max_steps}, Action: {action}")
                
                # Step the environment
                obs, reward, terminate, truncated, _ = env.step(action)
                
                print(f"Truncated: {truncated}, terminated: {terminate}, Reward: {reward}, Step: {step + 1}/{max_steps}")
                
                # Check for success or termination
                if terminate or truncated:
                    print(f"Episode {i} ended with {'success' if reward > 0 else 'failure'}")
                    if reward > 0:
                        success_rate += 1
                    break
                
            if step == max_steps - 1:
                print(f"Episode {i} reached max steps without termination.")
            else:
                print(f"Episode {i} terminated at step {step} with reward {reward}.")
            
            # Save video
            video_path = os.path.join(videos_dir, f"{task_name}_episode_{i}.mp4")
            env.tr.save(video_path, instruction)

        results[task_name] = {"success_rate": success_rate / n_episodes}
        
    # Save results
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    env.env.shutdown()

if __name__ == "__main__":
    test_openpi()