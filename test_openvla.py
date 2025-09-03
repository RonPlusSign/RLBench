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
from transformers import AutoModelForVision2Seq, AutoProcessor

from LIFT3D.lift3d.envs.rlbench_env import RLBenchActionMode, RLBenchObservationConfig, RLBenchEnv


def test_openvla():
    # Configuration
    tasks = [PutRubbishInBin, PutBooksOnBookshelf, EmptyContainer]
    checkpoint_path = "/home/adelli/RLBench/checkpoints/openvla-7b"
    results_dir = "runs/openvla_test"
    videos_dir = os.path.join(results_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # Load OpenVLA model
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)

    # Setup RLBench environment
    obs_config = RLBenchObservationConfig.multi_view_config(
        camera_names=["front", "wrist", "left_shoulder", "right_shoulder", "overhead"],
        image_size=(256, 256)
    )

    action_mode = RLBenchActionMode.eepose_then_gripper_action_mode(absolute=True)

    env = RLBenchEnv(
        task_name="put_rubbish_in_bin", # Will be changed in the loop
        action_mode=action_mode,
        obs_config=obs_config,
        headless=True,
        cinematic_record_enabled=False,
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
            # Print step information
            print(f"Observation: {obs}")               
            
            for step in range(max_steps):
                # Get action from OpenVLA
                image = Image.fromarray(obs['image'])
                prompt = f"In: What action should the robot take to {instruction}?\nOut:"
                inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

                # Predict the action
                # "maniskill_dataset_converted_externally_to_rlds" is the dataset in Open-X with the highest number of samples with Franka robot
                # Other options: 'viola', 'taco_play'
                action = vla.predict_action(**inputs, unnorm_key='toto', do_sample=False)
                
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
    test_openvla()
