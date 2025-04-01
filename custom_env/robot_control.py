import os
import sys
import numpy as np
import mujoco
import mujoco_viewer
from pynput import keyboard
import yaml

# Add parent directory to path to import simulator modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from simulator.robots import Draco3
from simulator.controllers import DracoController
from util import geom
from robosuite.utils.binding_utils import MjSim

class RobotController:
    def __init__(self, xml_path):
        # Convert relative path to absolute path
        xml_path = os.path.join(project_root, xml_path)
        
        # Load the model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Create simulation using robosuite's MjSim
        self.sim = MjSim(self.model)
        
        # Initialize robot
        self.robot = Draco3()
        
        # Load config
        with open(os.path.join(project_root, "configs/default.yaml"), "r") as file:
            self.config = yaml.safe_load(file)
            
        # Initialize controller
        self.controller = DracoController(self.config, os.path.join(project_root, "models/robots/draco3"))
        
        # Initialize state variables
        self.current_pose = "neutral"
        self.locomotion_mode = 0  # 0: balance, 1: walk_forward
        self.gripper_left = 0
        self.gripper_right = 0
        
        # Define poses
        self.poses = {
            "neutral": {
                "left_pos": np.array([0.22, 0.25, 0.1]),
                "right_pos": np.array([0.22, -0.25, 0.1]),
                "left_quat": np.array([1, 0, 0, 0]),
                "right_quat": np.array([1, 0, 0, 0])
            },
            "wave": {
                "left_pos": np.array([0.22, 0.25, 0.3]),
                "right_pos": np.array([0.22, -0.25, 0.1]),
                "left_quat": np.array([0.707, 0, 0, 0.707]),
                "right_quat": np.array([1, 0, 0, 0])
            },
            "reach": {
                "left_pos": np.array([0.5, 0.25, 0.3]),
                "right_pos": np.array([0.5, -0.25, 0.3]),
                "left_quat": np.array([1, 0, 0, 0]),
                "right_quat": np.array([1, 0, 0, 0])
            },
            "squat": {
                "left_pos": np.array([0.22, 0.25, 0.05]),
                "right_pos": np.array([0.22, -0.25, 0.05]),
                "left_quat": np.array([1, 0, 0, 0]),
                "right_quat": np.array([1, 0, 0, 0])
            }
        }
        
        # Set up keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        
    def on_press(self, key):
        try:
            if key.char == 'n':
                self.current_pose = "neutral"
            elif key.char == 'w':
                self.current_pose = "wave"
            elif key.char == 'r':
                self.current_pose = "reach"
            elif key.char == 's':
                self.current_pose = "squat"
            elif key.char == 'f':
                self.locomotion_mode = 1
            elif key.char == 'b':
                self.locomotion_mode = 0
            elif key.char == 'g':
                self.gripper_left = 1 - self.gripper_left
                self.gripper_right = 1 - self.gripper_right
            elif key.char == 'q':
                return False
        except AttributeError:
            pass
    
    def smooth_transition(self, start_pos, end_pos, start_quat, end_quat, t, duration=1.0):
        """Smoothly interpolate between two poses"""
        if t >= duration:
            return end_pos, end_quat
        
        # Linear interpolation for position
        alpha = t / duration
        pos = start_pos + alpha * (end_pos - start_pos)
        
        # Spherical interpolation for quaternion
        quat = geom.slerp(start_quat, end_quat, alpha)
        
        return pos, quat
    
    def step(self):
        # Get current pose
        current_pose = self.poses[self.current_pose]
        
        # Create action dictionary
        action = {
            "trajectory": {
                "left_pos": current_pose["left_pos"],
                "right_pos": current_pose["right_pos"],
                "left_quat": current_pose["left_quat"],
                "right_quat": current_pose["right_quat"]
            },
            "gripper": {
                "left": self.gripper_left,
                "right": self.gripper_right
            },
            "locomotion": self.locomotion_mode
        }
        
        # Update controller with action
        self.controller.update_trajectory(action["trajectory"], 
                                        "walk_forward" if self.locomotion_mode == 1 else "balance")
        self.controller.update_gripper_target(action["gripper"])
        
        # Get command from controller
        command = self.controller.get_command()
        
        # Apply command to simulation
        self.sim.data.ctrl[:] = command["joint_trq"].values()
        
        # Step simulation
        self.sim.step()
        
        return self.get_observation()
    
    def get_observation(self):
        return {
            'joint_positions': self.sim.data.qpos,
            'joint_velocities': self.sim.data.qvel,
            'pose': self.current_pose,
            'locomotion_mode': self.locomotion_mode,
            'gripper_left': self.gripper_left,
            'gripper_right': self.gripper_right
        }

def main():
    # Create controller
    controller = RobotController("custom_env/checkered.xml")
    
    # Create viewer
    viewer = mujoco_viewer.MujocoViewer(controller.model, controller.data)
    
    print("Controls:")
    print("n: Neutral pose")
    print("w: Wave pose")
    print("r: Reach pose")
    print("s: Squat pose")
    print("f: Walk forward")
    print("b: Balance mode")
    print("g: Toggle gripper")
    print("q: Quit")
    
    # Main loop
    while True:
        obs = controller.step()
        
        # Render
        if viewer.is_alive:
            viewer.render()
        else:
            break
    
    # Cleanup
    viewer.close()
    controller.listener.stop()

if __name__ == "__main__":
    main() 