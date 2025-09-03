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
import traceback
from tqdm import tqdm

# Import all tasks to make them available through globals()
from rlbench.tasks import *

def action_conversion(action: np.ndarray, to_representation: str = 'euler', is_relative: bool = False, previous_action: np.ndarray = None):
    """Convert an action between Euler and quaternion representations.

    Args:
        action: np.ndarray of shape (7,) (Euler) or (8,) (quaternion).
                Euler format: [x, y, z, roll, pitch, yaw, gripper]
                Quaternion format: [x, y, z, qx, qy, qz, qw, gripper]
        to_representation: 'euler' or 'quat' target representation.
        is_relative: if True, compute the delta between `action` and
                     `previous_action` and return the delta (position and
                     rotation). `previous_action` must be provided in this case.
        previous_action: previous action (same format as `action`) used when
                         is_relative is True.

    Returns:
        np.ndarray converted action in the target representation. For relative
        mode the returned rotation represents the delta rotation (as Euler
        angles or as a unit quaternion depending on `to_representation`).

    Notes:
        - Quaternion ordering is (qx, qy, qz, qw) to match the rest of the
          codebase. Rotation objects from scipy are created/consumed with
          this ordering via as_quat(scalar_first=False).
        - When producing quaternions we always normalize to guard against
          numerical drift.
    """
    if to_representation not in ('euler', 'quat'):
        raise ValueError("to_representation must be 'euler' or 'quat'")

    a = np.asarray(action, dtype=float)
    if a.size not in (7, 8):
        raise ValueError('action must be length 7 (Euler) or 8 (quaternion)')

    if is_relative and previous_action is None:
        raise ValueError('previous_action must be provided when is_relative is True')

    def _ensure_unit_quat(q):
        q = np.asarray(q, dtype=float)
        n = np.linalg.norm(q)
        if n == 0:
            raise ValueError('Zero quaternion encountered')
        return q / n

    # Helper: construct Rotation from either euler or quat stored in action array
    def _rot_from_action(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 7:
            return R.from_euler('xyz', arr[3:6], degrees=False)
        else:
            return R.from_quat(arr[3:7])  # (qx, qy, qz, qw)

    # Gripper state (keep as-is, demo code expects absolute gripper state even for deltas)
    gripper = a[-1]

    # Relative case: compute deltas
    if is_relative:
        prev = np.asarray(previous_action, dtype=float)
        if prev.size not in (7, 8):
            raise ValueError('previous_action must be length 7 or 8')

        delta_pos = a[:3] - prev[:3]

        # If both are Euler, simple subtraction of angles is fine
        if a.size == 7 and prev.size == 7:
            delta_ang = a[3:6] - prev[3:6]
            if to_representation == 'euler':
                return np.array([*delta_pos, *delta_ang, gripper], dtype=float)
            else:
                # convert delta Euler to quaternion (and normalize)
                q = R.from_euler('xyz', delta_ang, degrees=False).as_quat(scalar_first=False)
                q = _ensure_unit_quat(q)
                return np.array([*delta_pos, *q, gripper], dtype=float)

        # Otherwise use rotation algebra to compute the delta rotation
        r_cur = _rot_from_action(a)
        r_prev = _rot_from_action(prev)
        r_delta = r_cur * r_prev.inv()

        if to_representation == 'euler':
            delta_ang = r_delta.as_euler('xyz', degrees=False)
            return np.array([*delta_pos, *delta_ang, gripper], dtype=float)
        else:
            q = r_delta.as_quat(scalar_first=False)
            q = _ensure_unit_quat(q)
            return np.array([*delta_pos, *q, gripper], dtype=float)

    # Absolute case: just convert representations
    if to_representation == 'euler':
        if a.size == 7:
            return a.astype(float)
        else:
            euler = R.from_quat(a[3:7]).as_euler('xyz', degrees=False)
            return np.array([*a[:3], *euler, gripper], dtype=float)
    else:  # to_representation == 'quat'
        if a.size == 8:
            q = _ensure_unit_quat(a[3:7])
            return np.array([*a[:3], *q, gripper], dtype=float)
        else:
            q = R.from_euler('xyz', a[3:6], degrees=False).as_quat(scalar_first=False)
            q = _ensure_unit_quat(q)
            return np.array([*a[:3], *q, gripper], dtype=float)


def get_target_pose(demo: Demo, index: int):
    """ Get the target pose (gripper position and open state) for a specific observation in the demo. """
    return np.array([*demo._observations[max(0, index)].gripper_pose, demo._observations[max(0, index)].gripper_open])


def main(argv):
    # Dynamically get the task class
    try:
        task_class = globals()[FLAGS.task]
    except KeyError:
        raise ValueError(f"Task {FLAGS.task} not found.")

    # RLBench setup
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    # Choose action mode based on whether the demo is absolute or relative
    # If demo uses absolute poses, set absolute_mode=True so the arm controller
    # expects absolute end-effector poses each step. If the demo uses relative
    # (delta) poses, set absolute_mode=False.
    is_relative = FLAGS.positioning == 'relative'
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=not is_relative),
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
    current_target = get_target_pose(demo, 0)

    # Replay the demo actions
    for i, _ in tqdm(enumerate(demo.actions), total=len(demo.actions), desc="Replaying demo"):
        try:
            # Instead of using the recorded action, build the action from the recorded gripper pose + gripper open state (which are always absolute). 
            # How we do that depends on whether the demo is relative (deltas) or absolute (poses).
            previous_target = current_target
            current_target = get_target_pose(demo, i)

            action = action_conversion(current_target, 'quat', is_relative, previous_target)
            obs, reward, terminate = task.step(action)
        except Exception as e:
            print(f"Error during step {i+1}: {e}")
            traceback.print_exc()
            break

    print('Finished replaying demo.')
    env.shutdown()


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('demo_path', '', 'Path to the demo to replay.')
    flags.DEFINE_string("task", "PutRubbishInBin", "Name of the RLBench task.")
    flags.DEFINE_enum('positioning', 'relative', ['absolute', 'relative'], 'Whether the demo uses absolute poses or relative (delta) poses.')

    app.run(main)
