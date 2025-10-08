import os
import json
import torch
from PIL import Image
import argparse
from LIFT3D.lift3d.envs.rlbench_env import RLBenchEvaluator, RLBenchActionMode
from openpi.training import config
from openpi.shared import download
from openpi.policies import policy_config

class OpenPIPolicy(torch.nn.Module):
    """Adapter for OpenPI model to conform to RLBenchEvaluator policy interface."""
    def __init__(self, openpi_policy, device):
        super(OpenPIPolicy, self).__init__()
        self.openpi_policy = openpi_policy
        self.device = device
        # Dummy parameter to ensure parameters() is non-empty for evaluator device detection
        self.dummy_param = torch.nn.Parameter(torch.zeros(1, device=device))

    def forward(self, images, point_clouds, robot_states, texts):
        batch_size = images.shape[0]

        # print(f"Processing images with shape: {images.shape}")    # FIXME: <-------------------
        
        actions = []
        for i in range(batch_size):
            # convert tensor image to PIL image
            img = images[i].permute(1, 2, 0).cpu().byte().numpy()
            pil_img = Image.fromarray(img)
            
            # Save image to disk
            img_path = f"image_pi0_{i}.png"
            pil_img.save(img_path)
            print(f"Saved image {i} to {img_path}")
            
            # robot state to numpy
            state = robot_states[i].cpu().numpy()
            prompt = texts[i]
            openpi_obs = {
                "prompt": prompt,
                
                # LIBERO
                "observation/image": pil_img,
                "observation/state": state,
                
                # DROID
                # "observation/exterior_image_1_left": pil_img,
                # "observation/joint_position": state[:7].tolist(),
                # "observation/gripper_position": state[7:9].tolist(),
            }
            result = self.openpi_policy.infer(openpi_obs)
            action_chunk = result.get("actions")
            action = action_chunk[0]
            # print(f"Predicted action for batch {i}: {action}")    # FIXME: <-------------------
            
            actions.append(torch.tensor(action, dtype=torch.float32))
        return torch.stack(actions, dim=0).to(self.device)


def test_openpi(task_name, n_episodes):
    # Load base OpenPI model
    model_name = "pi0_libero"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    openpi_cfg = config.get_config(model_name)
    checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{model_name}")
    base_policy = policy_config.create_trained_policy(openpi_cfg, checkpoint_dir)
    policy = OpenPIPolicy(base_policy, device).to(device)

    # Setup RLBench evaluator
    print(f"Testing task: {task_name}")
    evaluator = RLBenchEvaluator(
        task_name=task_name,
        image_size=256,
        action_mode=RLBenchActionMode.eepose_then_gripper_action_mode(absolute=False),
        camera_name="front",
        point_cloud_camera_names=["front"],
        use_point_crop=True,
        num_points=1024,
        max_episode_length=100,
        rotation_representation='euler',   # droid=>quaternion, libero=>euler
        headless=True,
        verbose_warnings=True,
        cinematic_record_enabled=True,
    )

    # Run evaluation
    success_rate, avg_rewards = evaluator.evaluate(n_episodes, policy)
    print(f"Task {task_name}: success_rate={success_rate}, avg_rewards={avg_rewards}")

    # Save results
    results_dir = "runs/openpi_test"
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"results_{task_name}.json")
    with open(out_path, "w") as f:
        json.dump({task_name: {"success_rate": success_rate, "avg_rewards": avg_rewards}}, f, indent=4)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OpenPI policy on a specific RLBench task")
    parser.add_argument("task_name", type=str,
                        choices=["put_rubbish_in_bin", "put_books_on_bookshelf", "empty_container"],
                        help="Name of the RLBench task to test")
    parser.add_argument("n_episodes", type=int, default=10,
                        help="Number of episodes to run for evaluation")
    args = parser.parse_args()
    test_openpi(args.task_name, args.n_episodes)