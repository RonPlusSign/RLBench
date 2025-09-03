import numpy as np
from typing import List
from rlbench.backend.observation import Observation
import pickle
from enum import Enum
from scipy.spatial.transform import Rotation as R


# Enumeration containing action representations
class ActionRepresentation(Enum):
    # This defines how end-effector rotations are represented.
    # The action represents the end-effector pose: position, rotation, and gripper state (open/close).
    # It can be absolute or relative (delta from current pose).
    # The rotation can be represented as either Euler angles (3 values) or quaternions (4 values).
    EULER = "euler"  # Actions have 7 values: [x, y, z, roll, pitch, yaw, gripper_state]
    QUAT = "quat"  # Actions have 8 values: [x, y, z, qx, qy, qz, qw, gripper_state]


class Demo(object):

    def __init__(self, observations: List[Observation], random_seed=None, num_reset_attempts=None):
        self._observations = observations
        self.random_seed = random_seed
        self.num_reset_attempts = num_reset_attempts
        self.actions = []  # By default these are generated in quaternion representation (8 values), but can be converted to Euler (7 values) when saving.
        self.instruction = ""

    def __len__(self):
        return len(self._observations)

    def __getitem__(self, i) -> Observation:
        return self._observations[i]

    def save(self, file_path: str, action_representation: ActionRepresentation = ActionRepresentation.QUAT):
        """Save this demo instance to disk."""

        # Remove useless observation attributes ([left_shoulder*, right_shoulder*, overhead*, wrist*, front*] + [_depth, _mask, _point_cloud])
        for obs in self._observations:
            obs.left_shoulder_rgb = obs.left_shoulder_rgb.astype(np.uint8)
            obs.right_shoulder_rgb = obs.right_shoulder_rgb.astype(np.uint8)
            obs.overhead_rgb = obs.overhead_rgb.astype(np.uint8)
            obs.wrist_rgb = obs.wrist_rgb.astype(np.uint8)
            obs.front_rgb = obs.front_rgb.astype(np.uint8)

            obs.left_shoulder_depth = None
            obs.right_shoulder_depth = None
            obs.overhead_depth = None
            obs.wrist_depth = None
            obs.front_depth = None

            obs.left_shoulder_mask = None
            obs.right_shoulder_mask = None
            obs.overhead_mask = None
            obs.wrist_mask = None
            obs.front_mask = None

            obs.left_shoulder_point_cloud = None
            obs.right_shoulder_point_cloud = None
            obs.overhead_point_cloud = None
            obs.wrist_point_cloud = None
            obs.front_point_cloud = None

        # Convert actions to the desired action representation
        # Action format: Quaternion = [x, y, z, qx, qy, qz, qw, gripper], Euler = [x, y, z, roll, pitch, yaw, gripper]
        converted_actions = []
        if action_representation == ActionRepresentation.QUAT:  # Convert to quaternion representation if necessary
            for action in self.actions:
                if len(action) == 8:  # Already stored as quaternion (8 values)
                    converted_actions.append(action)
                    continue

                elif len(action) == 7:  # Convert Euler -> quaternion
                    pos = action[:3]
                    euler = action[3:6]
                    gripper = action[-1]
                    quat = R.from_euler("xyz", euler, degrees=False).as_quat(scalar_first=False) # Convert to quaternion (qx, qy, qz, qw) from Euler (xyz)
                    converted_actions.append(np.concatenate([pos, quat, [gripper]]))

        elif action_representation == ActionRepresentation.EULER:  # Convert to Euler representation if necessary
            for action in self.actions:
                if len(action) == 7: # Already stored as Euler (7 values)
                    converted_actions.append(action)
                    continue

                elif len(action) == 8:  # Convert quaternion -> Euler
                    pos = action[:3]
                    quat = action[3:7]  # [qx, qy, qz, qw]
                    gripper = action[-1]
                    euler = R.from_quat(quat).as_euler("xyz", degrees=False) # Convert to Euler (radians), convention: roll=x, pitch=y, yaw=z
                    converted_actions.append(np.concatenate([pos, euler, [gripper]]))
        self.actions = converted_actions

        # Save as a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path: str) -> "Demo":
        """Load a demo instance from disk."""
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def restore_state(self):
        np.random.set_state(self.random_seed)
