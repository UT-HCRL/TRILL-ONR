import os
import sys
import threading
from pynput import keyboard

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import mujoco
import mujoco_viewer
from util import geom
from simulator.envs import ShipEnv
from simulator.recorder import HDF5Recorder
import time
import argparse

RIGHTFORWARD_GRIPPER = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
TRANSFORM_VR = np.array(
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
)  # geom.euler_to_rot(np.array([0, np.pi, 0]))

# Global state for key presses
pressed_keys = set()

def on_press(key):
    try:
        pressed_keys.add(key.char)
    except AttributeError:
        pass

def on_release(key):
    try:
        pressed_keys.discard(key.char)
    except AttributeError:
        if key == keyboard.Key.esc:
            return False

def smooth_transition(start_pos, end_pos, start_rot, end_rot, t, duration):
    """Smoothly interpolate between start and end positions/rotations"""
    # Smooth easing function (cubic)
    t = np.clip(t / duration, 0, 1)
    t = t * t * (3 - 2 * t)
    
    # Interpolate positions
    pos = start_pos + (end_pos - start_pos) * t
    
    # Simple linear interpolation for rotations
    rot = start_rot + (end_rot - start_rot) * t
    
    # Normalize the rotation matrix to ensure it remains orthogonal
    u, s, vh = np.linalg.svd(rot)
    rot = u @ vh
    
    return pos, rot

def main(gui=1, save_video=False):
    env = ShipEnv()
    
    env.config["Manipulation"]["Trajectory Mode"] = "interpolation"
    
    # Get the native mujoco model and data
    model = env.sim.model._model
    data = env.sim.data._data
    
    # Create the viewer if GUI is enabled
    viewer = None
    if gui:
        viewer = mujoco_viewer.MujocoViewer(model, data)
    
    # Set up recorder if needed
    recorder = None
    if save_video:
        recorder = HDF5Recorder(
            sim=env.sim,
            config=env.config,
            file_path=f"./test/demo_robot_{int(time.time())}",
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
    next_pose = poses['neutral']
    next_pose_name = 'neutral'
    transition_duration = 1.5  # Increased duration for smoother transitions
    action_start_time = env.cur_time
    current_time = 0
    phase = 0
    locomotion_mode = 0  # 0: balance, 1: walk_forward
    
    # Gripper state
    gripper_state = 0  # 0: open, 1: closed
    gripper_transition_duration = 0.5
    gripper_start_time = env.cur_time
    gripper_current_time = 0

    # Print instructions
    print("\nKeyboard Controls:")
    print("1: Neutral pose")
    print("2: Wave")
    print("3: Reach")
    print("4: Squat")
    print("W: Walk forward")
    print("S: Stop walking/Balance mode")
    print("A: Strafe left")
    print("D: Strafe right")
    print("Q: Turn left")
    print("E: Turn right")
    print("G: Toggle gripper")
    print("ESC: Quit")

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    while not done:
        action = {}
        action["trajectory"] = {}
        action["gripper"] = {}
        action["aux"] = {}
        action["subtask"] = 0
        action["locomotion"] = locomotion_mode

        # Handle keyboard input
        if '1' in pressed_keys:
            current_pose = next_pose
            current_pose_name = next_pose_name
            next_pose = poses['neutral']
            next_pose_name = 'neutral'
            action_start_time = env.cur_time
            current_time = 0
        elif '2' in pressed_keys:
            current_pose = next_pose
            current_pose_name = next_pose_name
            next_pose = poses['wave']
            next_pose_name = 'wave'
            action_start_time = env.cur_time
            current_time = 0
        elif '3' in pressed_keys:
            current_pose = next_pose
            current_pose_name = next_pose_name
            next_pose = poses['reach']
            next_pose_name = 'reach'
            action_start_time = env.cur_time
            current_time = 0
        elif '4' in pressed_keys:
            current_pose = next_pose
            current_pose_name = next_pose_name
            next_pose = poses['squat']
            next_pose_name = 'squat'
            action_start_time = env.cur_time
            current_time = 0
        elif 'w' in pressed_keys:
            locomotion_mode = 1  # walk_forward
        elif 's' in pressed_keys:
            locomotion_mode = 0  # balance
        elif 'a' in pressed_keys:
            locomotion_mode = 3  # strafe_left
        elif 'd' in pressed_keys:
            locomotion_mode = 4  # strafe_right
        elif 'q' in pressed_keys:
            locomotion_mode = 5  # turn_left
        elif 'e' in pressed_keys:
            locomotion_mode = 6  # turn_right
            print(f"Pressed {pressed_keys}")
        elif 'x' in pressed_keys:
            locomotion_mode = 8  # walk_in_x
        elif 'g' in pressed_keys:
            gripper_state = 1 - gripper_state  # Toggle between 0 and 1
            gripper_start_time = env.cur_time
            gripper_current_time = 0
        elif keyboard.Key.esc in pressed_keys:
            done = True

        # Update time
        current_time = env.cur_time - action_start_time
        gripper_current_time = env.cur_time - gripper_start_time

        # Smooth transition between poses
        if current_time < transition_duration:
            # During transition
            rh_pos, rh_rot = smooth_transition(
                current_pose['right_pos'],
                next_pose['right_pos'],
                current_pose['right_rot'],
                next_pose['right_rot'],
                current_time,
                transition_duration
            )
            lh_pos, lh_rot = smooth_transition(
                current_pose['left_pos'],
                next_pose['left_pos'],
                current_pose['left_rot'],
                next_pose['left_rot'],
                current_time,
                transition_duration
            )
        else:
            # After transition, maintain final pose
            rh_pos = next_pose['right_pos']
            rh_rot = next_pose['right_rot']
            lh_pos = next_pose['left_pos']
            lh_rot = next_pose['left_rot']

        # Add walking motion
        phase = (env.cur_time - init_time) / 6.0
        # if locomotion_mode == 1:  # walk_forward mode
        #     # Add walking motion to arms
        #     walk_offset = 0.2 * np.sin(2 * np.pi * phase) * np.array([-1.0, 0.0, -1.3])
        #     lh_pos += walk_offset
        #     rh_pos += walk_offset
        # elif next_pose_name == 'squat':
        #     # Add squat motion to arms with more pronounced oscillation
        #     squat_phase = np.sin(2 * np.pi * phase)
        #     squat_offset = 0.8 * np.array([-0.5, 0.0, -1.0]) + 0.1 * squat_phase * np.array([0.0, 0.0, 1.0])
        #     lh_pos += squat_offset
        #     rh_pos += squat_offset

        # Apply rotations and set positions
        rh_target_rot = np.dot(rh_rot, RIGHTFORWARD_GRIPPER)
        lh_target_rot = np.dot(lh_rot, RIGHTFORWARD_GRIPPER)
        action["trajectory"]["left_pos"] = lh_pos
        action["trajectory"]["right_pos"] = rh_pos
        action["trajectory"]["right_quat"] = geom.rot_to_quat(rh_target_rot)
        action["trajectory"]["left_quat"] = geom.rot_to_quat(lh_target_rot)

        # Handle gripper control
        if gripper_current_time < gripper_transition_duration:
            # Smooth transition for grippers
            t = np.clip(gripper_current_time / gripper_transition_duration, 0, 1)
            t = t * t * (3 - 2 * t)  # Cubic easing
            gripper_value = t if gripper_state == 1 else (1 - t)
        else:
            gripper_value = gripper_state

        action["gripper"]["left"] = gripper_value
        action["gripper"]["right"] = gripper_value

        obs = env.step(action)
        # print(obs)

        # Render if viewer is active
        if viewer and viewer.is_alive:
            viewer.render()
        elif viewer:
            break

        # Small delay to prevent overwhelming the system
        time.sleep(0.01)

    # Cleanup
    if viewer:
        viewer.close()
    if recorder:
        recorder.close()
    listener.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=int, default=1, help="Enable/disable GUI")
    parser.add_argument("--save_video", action="store_true", help="Save video of the simulation")
    args = parser.parse_args()

    main(gui=args.gui, save_video=args.save_video) 