import numpy as np
from absl import app
from absl import flags
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.demo import Demo
from scipy.spatial.transform import Rotation as R

# Import all tasks to make them available through globals()
from rlbench.tasks import *

FLAGS = flags.FLAGS
flags.DEFINE_string('demo_path', '', 'Path to the demo to replay.')
flags.DEFINE_string("task", "PutRubbishInBin", "Name of the RLBench task.")


def main(argv):
    # Dynamically get the task class
    try:
        task_class = globals()[FLAGS.task]
    except KeyError:
        raise ValueError(f"Task {FLAGS.task} not found.")

    # RLBench setup
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    # Use absolute actions since we are replaying a demo
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=False),
        gripper_action_mode=Discrete(),
    )
    env = Environment(action_mode, obs_config=obs_config, headless=False)
    env.launch()

    task = env.get_task(task_class)

    # Load the demo
    try:
        demo = Demo.load(FLAGS.demo_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Demo file not found at {FLAGS.demo_path}")

    # Reset task to the demo's initial state
    descriptions, obs = task.reset_to_demo(demo)
    print("Reset to demo initial state.")
    print("Instructions: ", descriptions)

    print(f"Actions: {demo.actions}")

    # Replay the demo actions
    for i, action in enumerate(demo.actions):
        print(f"Step {i+1}/{len(demo.actions)}, playing action: {action}")
        try:

            # If the length of the action is 7, it's in Euler and I need to convert it to quaternion
            if len(action) == 7:
                # Action is: [x y z roll pitch yaw gripper_state]
                # Convert rotation to quaternion: [x y z qx qy qz qw gripper_state]
                roll, pitch, yaw = action[-4:-1]
                quat = R.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_quat(scalar_first=False) # Convert to quaternion (qx, qy, qz, qw) from Euler (xyz)
                action = [*action[:3], *quat, action[-1]]
            else:
                # If the length of the action is not 7, it's already in quaternion
                # format: [x y z qx qy qz qw gripper_state]
                pass

            obs, reward, terminate = task.step(action)
        except Exception as e:
            print(f"Error during step {i+1}: {e}")
            break

    print('Finished replaying demo.')
    env.shutdown()


if __name__ == '__main__':
    app.run(main)
