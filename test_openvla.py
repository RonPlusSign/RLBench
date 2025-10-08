import os
import json
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import argparse
from LIFT3D.lift3d.envs.rlbench_env import RLBenchEvaluator, RLBenchActionMode
import numpy as np
import wandb

class OpenVLAPolicy(torch.nn.Module):
    """Adapter for OpenVLA model to conform to RLBenchEvaluator policy interface."""
    def __init__(self, processor, model, device):
        super(OpenVLAPolicy, self).__init__()
        self.processor = processor
        self.model = model
        self.device = device

    def forward(self, images, point_clouds, robot_states, texts):
        """Predict action given observations and text prompt """
        
        # Convert tensor images [B, C, H, W] to PIL images
        batch_size = images.shape[0]
        pil_images = []
        # print(f"Images shape: {images.shape}")
        for i in range(batch_size):
            img = images[i].permute(1, 2, 0).cpu().numpy().astype('uint8')
            pil_images.append(Image.fromarray(img))

        # Form prompts
        prompts = [f"In: What action should the robot take to {text}?\nOut:" for text in texts]

        # Tokenize and process images
        inputs = self.processor(prompts, pil_images, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(self.device) for k, v in inputs.items()}

        # Predict action
        action = self.model.predict_action(**inputs, unnorm_key='PutRubbishInBin_euler_relative', do_sample=False) # TODO: CHANGE THIS BASED ON DATASET USED FOR FINETUNING

        # Ensure tensor output on CPU
        # if isinstance(action, torch.Tensor):
        #     return action.cpu()
        return torch.tensor(action)


def test_openvla(task_name, n_episodes):
    # Configuration
    # checkpoint_path = "openvla/openvla-7b"
    # checkpoint_path = "/home/adelli/RLBench/checkpoints/openvla-7b"
    # checkpoint_path = "/home/adelli/openvla/checkpoints/openvla-7b+PutRubbishInBin_relative_euler+b10+lr-0.0005+lora-r32+dropout-0.0--image_aug"
    # checkpoint_path = "/home/adelli/openvla/checkpoints/openvla-7b+PutRubbishInBin_absolute_euler+b10+lr-0.0005+lora-r32+dropout-0.0--image_aug"
    checkpoint_path = "/home/adelli/openvla/checkpoints/openvla-7b+PutRubbishInBin_euler_relative+b16+lr-0.0005+lora-r32+dropout-0.0"
    results_dir = "runs/openvla_test"
    os.makedirs(results_dir, exist_ok=True)

    # Load OpenVLA model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        # attn_implementation="flash_attention_2",
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

    policy = OpenVLAPolicy(processor, vla, device)
    max_steps = 300 if task_name == "put_rubbish_in_bin" else (400 if task_name == "put_books_on_bookshelf" else 1000)

    os.environ['LIBGL_ALWAYS_SOFTWARE'] = 'true'

    results = {}
    print(f"Testing task: {task_name}")
    evaluator = RLBenchEvaluator(
        task_name=task_name,
        image_size=256,
        action_mode=RLBenchActionMode.eepose_then_gripper_action_mode(absolute=False),
        camera_name="left_shoulder",
        point_cloud_camera_names=["overhead", "wrist"],
        use_point_crop=True,
        num_points=1024,
        max_episode_length=max_steps,
        rotation_representation='euler',
        headless=True,
        verbose_warnings=True,
        cinematic_record_enabled=False,
        require_video_wrapper=True
    )
    success_rate, avg_rewards = evaluator.evaluate(n_episodes, policy, verbose=True, verbose_with_cinematic=False)
    evaluator.callback_verbose(wandb_logger=wandb)
    # evaluator.callback(logging_info=wandb)
    print(f"Task {task_name}: success_rate={success_rate}, avg_rewards={avg_rewards}")
    results[task_name] = {"success_rate": success_rate, "avg_rewards": avg_rewards}

    # Save results
    with open(os.path.join(results_dir, f"results_{task_name}.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test OpenVLA policy on a specific RLBench task")
    parser.add_argument("task_name", type=str, choices=["put_rubbish_in_bin", "put_books_on_bookshelf", "empty_container"], help="Name of the RLBench task to test")
    parser.add_argument("n_episodes", type=int, default=10, help="Number of episodes to run for evaluation")
    args = parser.parse_args()
    wandb.init(project="vla-eval", entity="andrea-delli-politecnico-di-torino", name=f"openvla-{args.task_name}")
    test_openvla(args.task_name, args.n_episodes)