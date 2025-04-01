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

import util.geom as geom

def main(gui=1, save_video=False):
    # Load the checkered environment XML
    model = mujoco.MjModel.from_xml_path("custom_env/checkered.xml")
    data = mujoco.MjData(model)
    
    # Create the viewer if GUI is enabled
    viewer = None
    if gui:
        viewer = mujoco_viewer.MujocoViewer(model, data)
        viewer.cam.distance = 5.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90
    
    # Initialize time
    init_time = time.time()
    
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
    done = False
    while not done:
        # Get current time
        cur_time = time.time() - init_time
        
        # Add walking motion if in walk mode
        if locomotion_mode == 1:
            phase = cur_time / 6.0
            walk_offset = 0.2 * np.sin(2 * np.pi * phase) * np.array([-1.0, 0.0, -1.3])
            current_pose['left_pos'] += walk_offset
            current_pose['right_pos'] += walk_offset

        # Set robot end-effector positions and orientations
        # Note: You'll need to map these to the correct joint indices in your XML
        # This is just an example - adjust the indices based on your XML structure
        right_arm_joint_indices = [model.joint("joint_right_S0").id, 
                                 model.joint("joint_right_S1").id,
                                 model.joint("joint_right_E0").id,
                                 model.joint("joint_right_E1").id,
                                 model.joint("joint_right_W0").id,
                                 model.joint("joint_right_W1").id,
                                 model.joint("joint_right_W2").id]
        
        left_arm_joint_indices = [model.joint("joint_left_S0").id,
                                model.joint("joint_left_S1").id,
                                model.joint("joint_left_E0").id,
                                model.joint("joint_left_E1").id,
                                model.joint("joint_left_W0").id,
                                model.joint("joint_left_W1").id,
                                model.joint("joint_left_W2").id]

        # Set gripper positions
        if gripper_left:
            data.actuator("gripper_left").ctrl = 1.0
        else:
            data.actuator("gripper_left").ctrl = 0.0
            
        if gripper_right:
            data.actuator("gripper_right").ctrl = 1.0
        else:
            data.actuator("gripper_right").ctrl = 0.0

        # Step the simulation
        mujoco.mj_step(model, data)

        # Render if GUI is enabled
        if viewer and viewer.is_alive:
            viewer.render()
        else:
            done = True

        # Small delay to prevent overwhelming the system
        time.sleep(0.01)

    # Cleanup
    if viewer:
        viewer.close()
    listener.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=int, default=1, help="Enable GUI")
    parser.add_argument("--save_video", action="store_true", help="Save video")
    args = parser.parse_args()

    main(gui=args.gui, save_video=args.save_video) 