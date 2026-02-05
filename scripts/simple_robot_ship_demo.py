import os
import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco_viewer
from pynput import keyboard

# Add parent directory to path to import simulator modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from simulator.envs.ship import ShipEnv
from simulator.recorder import HDF5Recorder
import util.geom as geom

def main(gui=1, save_video=False):
    # Initialize ship environment
    env = ShipEnv()
    
    env.config["Manipulation"]["Trajectory Mode"] = "interpolation"
    
    # Get the native mujoco model and data
    model = env.sim.model._model
    data = env.sim.data._data
    
    # Create the viewer if GUI is enabled
    viewer = None
    if gui:
        viewer = mujoco_viewer.MujocoViewer(model, data)
        viewer.cam.distance = env.camera_params["distance"]
        viewer.cam.elevation = env.camera_params["elevation"]
        viewer.cam.azimuth = env.camera_params["azimuth"]
    
    # Set up recorder if needed
    recorder = None
    if save_video:
        recorder = HDF5Recorder(
            sim=env.sim,
            config=env.config,
            file_path=f"./test/demo_robot_ship_{int(time.time())}",
        )
        env.set_recorder(recorder)
    
    env.reset()
    done = False
    init_time = env.cur_time

    # Base positions for arms
    right_pos = np.array([0.22, -0.25, 0.1])
    left_pos = np.array([0.22, 0.25, 0.1])

    # Define different poses and actions
    poses = {
        'neutral': {
            'right_pos': right_pos,
            'left_pos': left_pos,
            'right_rot': np.eye(3),
            'left_rot': np.eye(3)
        },
        'wave': {
            'right_pos': np.array([0.4, -0.3, 0.3]),
            'left_pos': np.array([0.4, 0.3, 0.3]),
            'right_rot': geom.euler_to_rot(np.array([0, 0, np.pi/4])),
            'left_rot': geom.euler_to_rot(np.array([0, 0, -np.pi/4]))
        },
        'reach': {
            'right_pos': np.array([0.6, -0.2, 0.4]),
            'left_pos': np.array([0.6, 0.2, 0.4]),
            'right_rot': geom.euler_to_rot(np.array([0, np.pi/4, 0])),
            'left_rot': geom.euler_to_rot(np.array([0, -np.pi/4, 0]))
        },
        'squat': {
            'right_pos': np.array([0.22, -0.25, -0.6]),  # Much lower position
            'left_pos': np.array([0.22, 0.25, -0.6]),   # Much lower position
            'right_rot': geom.euler_to_rot(np.array([0, 0, 0])),
            'left_rot': geom.euler_to_rot(np.array([0, 0, 0]))
        }
    }

    # Initialize state variables
    current_pose = poses['neutral']
    current_pose_name = 'neutral'
    locomotion_mode = 0  # 0: balance, 1: walk_forward
    gripper_left = 0
    gripper_right = 0

    def on_press(key):
        nonlocal current_pose, current_pose_name, locomotion_mode, gripper_left, gripper_right
        try:
            if key.char == 'n':
                current_pose = poses['neutral']
                current_pose_name = 'neutral'
            elif key.char == 'w':
                current_pose = poses['wave']
                current_pose_name = 'wave'
            elif key.char == 'r':
                current_pose = poses['reach']
                current_pose_name = 'reach'
            elif key.char == 's':
                current_pose = poses['squat']
                current_pose_name = 'squat'
            elif key.char == 'f':
                locomotion_mode = 1
            elif key.char == 'b':
                locomotion_mode = 0
            elif key.char == 'g':
                gripper_left = 1 - gripper_left
                gripper_right = 1 - gripper_right
            elif key.char == 'q':
                return False
        except AttributeError:
            pass

    # Set up keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("Controls:")
    print("n: Neutral pose")
    print("w: Wave pose")
    print("r: Reach pose")
    print("s: Squat pose")
    print("f: Walk forward")
    print("b: Balance mode")
    print("g: Toggle gripper")
    print("q: Quit")

    # Main simulation loop
    while not done:
        # Create action dictionary
        action = {
            "trajectory": {
                "left_pos": current_pose["left_pos"],
                "right_pos": current_pose["right_pos"],
                "left_quat": geom.rot_to_quat(current_pose["left_rot"]),
                "right_quat": geom.rot_to_quat(current_pose["right_rot"])
            },
            "gripper": {
                "left": gripper_left,
                "right": gripper_right
            },
            "locomotion": locomotion_mode
        }

        # Add walking motion if in walk mode
        if locomotion_mode == 1:
            phase = (env.cur_time - init_time) / 6.0
            walk_offset = 0.2 * np.sin(2 * np.pi * phase) * np.array([-1.0, 0.0, -1.3])
            action["trajectory"]["left_pos"] = current_pose["left_pos"] + walk_offset
            action["trajectory"]["right_pos"] = current_pose["right_pos"] + walk_offset

        # Step the environment
        obs = env.step(action)

        # Render if GUI is enabled
        if viewer and viewer.is_alive:
            viewer.render()
        else:
            done = True

    # Cleanup
    if viewer:
        viewer.close()
    if recorder:
        recorder.close()
    listener.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=int, default=1, help="Enable GUI")
    parser.add_argument("--save_video", action="store_true", help="Save video")
    args = parser.parse_args()

    main(gui=args.gui, save_video=args.save_video) 