import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
from util import geom
from simulator.envs import DoorEnv, EmptyEnv
from simulator.render import CV2Renderer
from simulator.recorder import HDF5Recorder
import time
import argparse

from simulator import sim_util

def open_door(env, lh_target_pos, lh_input_quat, gain=0.02, stage=0):

    if env.door is None:
        return False
    grasping_state = sim_util.get_grasping_state(env.sim, env.robot, env.door)

    if stage == 0:
        task = "grasping"
        tolerance = 0.01
        dims = [0, 1, 2]
        grasping = False
    elif stage == 1:
        task = "rotating"
        tolerance = 0.02
        dims = [0, 2]
        grasping = True
    else:
        task = "releasing"
        tolerance = 0.02
        dims = [0]
        grasping = True

    target_pos = grasping_state[f"latch_{task}_pos"]
    lh_eef_pos = grasping_state["lh_grasping_pos"]


    error_pos = target_pos - lh_eef_pos
    new_lh_target_pos = lh_target_pos + gain * error_pos

    if np.linalg.norm(error_pos[dims]) < tolerance:
        done = True
    else:
        done = False

    return new_lh_target_pos, lh_input_quat, grasping, done


def reset_hand(env, cur_target_pos, input_quat, gain=0.02, tolerance=0.01, hand="left"):

    if hand == "left":
        reset_pos = np.array([0.22, 0.25, 0.1])
    else:
        reset_pos = np.array([0.22, -0.25, 0.1])

    input_quat = geom.euler_to_quat(np.array([0, 0, 0]))

    error_pos = reset_pos - cur_target_pos
    new_target_pos = cur_target_pos + gain * error_pos

    if np.linalg.norm(error_pos) < tolerance:
        done = True
    else:
        done = False

    return new_target_pos, input_quat, False, done


def push_door(env, rh_target_pos, rh_input_quat, gain=0.02, tolerance=0.01, hand="left"):

    target_pos = np.array([0.38, -0.2, 0.1])


    error_pos = target_pos - rh_target_pos
    new_target_pos = rh_target_pos + gain * error_pos

    if np.linalg.norm(error_pos) < tolerance:
        done = True
    else:
        done = False

    return new_target_pos, rh_input_quat, False, done


RIGHTFORWARD_GRIPPER = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
TRANSFORM_VR = np.array(
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
)  # geom.euler_to_rot(np.array([0, np.pi, 0]))

ENV_LOOKUP = {
    "door": DoorEnv,
}


def main(gui, env_type, cam_name="upview", subtask=0, save_video=False):
    if env_type in ENV_LOOKUP.keys():
        env_class = ENV_LOOKUP[env_type]
    else:
        env_class = EmptyEnv

    env = env_class()

    env.config["Manipulation"]["Trajectory Mode"] = "interpolation"

    if save_video:
        save_path = os.path.join(
            ".", "{}_{}_{}.mp4".format(env_type, cam_name, subtask)
        )
    else:
        save_path = None
    renderer = CV2Renderer(
        device_id=-1, sim=env.sim, cam_name=cam_name, gui=gui, save_path=save_path
    )
    recorder = None
    recorder = HDF5Recorder(
        sim=env.sim,
        config=env.config,
        file_path="./test/demo_{}_{}".format(env_type, int(time.time())),
    )

    env.set_renderer(renderer)
    env.set_recorder(recorder)
    env.reset(subtask=subtask)

    done = False
    subtask = 0

    init_time = env.cur_time

    right_pos = np.array([0.22, -0.25, 0.1])
    left_pos = np.array([0.22, 0.25, 0.1])

    lh_target_pos = left_pos
    lh_input_quat = geom.euler_to_quat(np.array([0, 0, 0]))

    rh_target_pos = right_pos
    rh_input_quat = geom.euler_to_quat(np.array([0, 0, 0]))

    stage = 0

    while not done:
        action = {}
        action["trajectory"] = {}
        action["gripper"] = {}
        action["aux"] = {}
        action["subtask"] = 0
        action["locomotion"] = 0

        if stage < 3:
            lh_target_pos, lh_input_quat, grasping, task_done = open_door(env, lh_target_pos, lh_input_quat, gain=0.02, stage=stage)
            stage += task_done
        if stage == 3:
            lh_target_pos, lh_input_quat, grasping, task_done = reset_hand(env, lh_target_pos, lh_input_quat, gain=0.02, hand="left")
            stage += task_done
        if stage == 4:
            # this stage should be with walking
            rh_target_pos, rh_input_quat, grasping, task_done = push_door(env, rh_target_pos, rh_input_quat, gain=0.02)
            stage += task_done
        if stage == 5:
            rh_target_pos, rh_input_quat, grasping, task_done = reset_hand(env, rh_target_pos, rh_input_quat, gain=0.02, hand="right")

        print("Stage:", stage)

        lh_input = geom.quat_to_rot(lh_input_quat)
        rh_input = geom.quat_to_rot(rh_input_quat)

        rh_target_rot = np.dot(rh_input, RIGHTFORWARD_GRIPPER)
        lh_target_rot = np.dot(lh_input, RIGHTFORWARD_GRIPPER)
        action["trajectory"]["left_pos"] = lh_target_pos
        action["trajectory"]["right_pos"] = rh_target_pos
        action["trajectory"]["right_quat"] = geom.rot_to_quat(rh_target_rot)
        action["trajectory"]["left_quat"] = geom.rot_to_quat(lh_target_rot)
        action["gripper"]["left"] = grasping
        action["gripper"]["right"] = False

        obs = env.step(action)

        if env.cur_time > 100.0:
            done = True

    recorder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=int, default=1, help="")
    parser.add_argument("--env", type=str, default="door", help="")
    parser.add_argument("--cam", type=str, default="upview", help="")
    parser.add_argument("--subtask", type=int, default=1, help="")
    args = parser.parse_args()

    gui = args.gui
    env_type = args.env
    cam_name = args.cam
    subtask = args.subtask

    main(gui=gui, env_type=env_type, cam_name=cam_name, subtask=subtask)