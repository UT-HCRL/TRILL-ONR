import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import mujoco
import mujoco_viewer
from util import geom
from simulator.envs import CheckeredEnv
from simulator.recorder import HDF5Recorder
import time
import argparse

RIGHTFORWARD_GRIPPER = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
TRANSFORM_VR = np.array(
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
)  # geom.euler_to_rot(np.array([0, np.pi, 0]))

def main(gui=1, save_video=False):
    env = CheckeredEnv()
    
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
            file_path=f"./test/demo_checkered_{int(time.time())}",
        )
        env.set_recorder(recorder)
    
    env.reset()
    done = False
    init_time = env.cur_time

    right_pos = np.array([0.22, -0.25, 0.1])
    left_pos = np.array([0.22, 0.25, 0.1])

    while not done:
        action = {}
        action["trajectory"] = {}
        action["gripper"] = {}
        action["aux"] = {}
        action["subtask"] = 0
        action["locomotion"] = 0

        rh_target_pos = right_pos
        lh_target_pos = left_pos
        lh_input = geom.euler_to_rot(np.array([0, 0, 0]))
        rh_input = geom.euler_to_rot(np.array([0, 0, 0]))

        if env.cur_time < 3.0 + init_time:
            if env.cur_time < 1.5 + init_time:
                phase = (env.cur_time - init_time) / 6.0
            else:
                phase = 0.25
            lh_target_pos = left_pos + 0.2 * np.sin(2 * np.pi * phase) * np.array(
                [-1.0, 0.0, -1.3]
            )
            rh_target_pos = right_pos + 0.2 * np.sin(2 * np.pi * phase) * np.array(
                [-1.0, 0.0, -1.3]
            )
            action["locomotion"] = 0
        else:
            lh_target_pos = left_pos + 0.4 * np.array([-0.5, 0.0, -1.0])
            rh_target_pos = right_pos + 0.4 * np.array([-0.5, 0.0, -1.0])
            action["locomotion"] = 1

        rh_target_rot = np.dot(rh_input, RIGHTFORWARD_GRIPPER)
        lh_target_rot = np.dot(lh_input, RIGHTFORWARD_GRIPPER)
        action["trajectory"]["left_pos"] = lh_target_pos
        action["trajectory"]["right_pos"] = rh_target_pos
        action["trajectory"]["right_quat"] = geom.rot_to_quat(rh_target_rot)
        action["trajectory"]["left_quat"] = geom.rot_to_quat(lh_target_rot)

        obs = env.step(action)
        print(obs)

        # Render if viewer is active
        if viewer and viewer.is_alive:
            viewer.render()
        elif viewer:
            break

        if env.cur_time > 24.0:
            done = True

    # Cleanup
    if viewer:
        viewer.close()
    if recorder:
        recorder.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=int, default=1, help="Enable/disable GUI")
    parser.add_argument("--save_video", action="store_true", help="Save video of the simulation")
    args = parser.parse_args()

    main(gui=args.gui, save_video=args.save_video) 