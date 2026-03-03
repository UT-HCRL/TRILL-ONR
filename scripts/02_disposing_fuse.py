import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
from util import geom
from simulator.envs import EmptyEnv, ShipEnv
from simulator.render import CV2Renderer
from simulator.recorder import HDF5Recorder
import time
import argparse

from simulator import sim_util

import mujoco_viewer

def pick_used_socket(env, lh_target_pos, lh_input_quat, gain=0.02, stage=0):

    if env.used_socket is None:
        return False
    grasping_state = sim_util.get_grasping_state(env.sim, env.robot, env.used_socket)

    if stage == 0:
        task = "grasping"
        target_pos = grasping_state[f"latch_{task}_pos"]
        tolerance = 0.015
        dims = [0, 1, 2]
        grasping = False
    else:
        task = "releasing"
        target_pos = np.array([0.3, 0.3, 0.15])
        tolerance = 0.02
        dims = [0, 1, 2]
        grasping = True

    lh_eef_pos = grasping_state["lh_grasping_pos"]

    error_pos = target_pos - lh_eef_pos
    new_lh_target_pos = lh_target_pos + gain * error_pos

    print(error_pos)

    if np.linalg.norm(error_pos[dims]) < tolerance:
        done = True
    else:
        done = False

    return new_lh_target_pos, lh_input_quat, grasping, done


def discard_used_socket(env, lh_target_pos, lh_input_quat, gain=0.02, stage=0):

    if env.used_socket is None:
        return False
    grasping_state = sim_util.get_grasping_state(env.sim, env.robot, env.trash_can)

    if stage == 4:
        task = "socket"
        target_pos = grasping_state[f"latch_{task}_pos"]
        tolerance = 0.05
        dims = [0, 1]
        grasping = True
    else:
        task = "releasing"
        target_pos = np.array([0.4, 0.4, 0.1])
        tolerance = 0.05
        dims = [0, 1]
        grasping = False

    lh_eef_pos = grasping_state["lh_grasping_pos"]

    error_pos = target_pos - lh_eef_pos
    new_lh_target_pos = lh_target_pos + gain * error_pos
    print(np.linalg.norm(error_pos[dims]) ,tolerance)

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


def slide_door(env, rh_target_pos, rh_input_quat, gain=0.02, tolerance=0.01, hand="left"):

    target_pos = np.array([0.12, 0.25, 0.1])


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
    "door": ShipEnv,
}


def main(gui, env_type, cam_name="upview", subtask=-1, save_video=True):
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

    # Get the native mujoco model and data
    model = env.sim.model._model
    data = env.sim.data._data
    # Create the viewer if GUI is enabled
    viewer = None
    if gui:
        viewer = mujoco_viewer.MujocoViewer(model, data)
    
    # renderer = CV2Renderer(
    #     device_id=-1, sim=env.sim, cam_name=cam_name, gui=gui, save_path=save_path
    # )
    renderer = None
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

    init_time = env.cur_time

    right_pos = np.array([0.22, -0.25, 0.1])
    left_pos = np.array([0.22, 0.25, 0.1])

    lh_target_pos = left_pos
    lh_input_quat = geom.euler_to_quat(np.array([0.5*np.pi, 0, 0]))

    rh_target_pos = right_pos
    rh_input_quat = geom.euler_to_quat(np.array([-0.0*np.pi, 0, 0]))

    stage = 0
    walking_cnt = 0

    while not done:
        action = {}
        action["trajectory"] = {}
        action["gripper"] = {}
        action["aux"] = {}
        action["subtask"] = subtask
        action["locomotion"] = 0
        print(stage)

        if stage < 2:
            lh_target_pos, lh_input_quat, grasping, task_done = pick_used_socket(env, lh_target_pos, lh_input_quat, gain=0.02, stage=stage)
            stage += task_done

        if stage == 3:
            walking_cnt += 1
            if walking_cnt > 400:
                task_done = 1
                walking_cnt = 0
            else:
                task_done = 0
                action["locomotion"] = 4
            stage += task_done

        if stage == 2:
            walking_cnt += 1
            if walking_cnt > 200:
                task_done = 1
                walking_cnt = 0
            else:
                task_done = 0
                action["locomotion"] = 2
            stage += task_done

        if stage < 6 and stage >=4:
            lh_target_pos, lh_input_quat, grasping, task_done = discard_used_socket(env, lh_target_pos, lh_input_quat, gain=0.02, stage=stage)
            stage += task_done


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

        if viewer and viewer.is_alive:
            viewer.render()
        elif viewer:
            break

        # # Small delay to prevent overwhelming the system
        # time.sleep(0.01)

    # Cleanup
    if viewer:
        viewer.close()

    recorder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=int, default=1, help="")
    parser.add_argument("--env", type=str, default="door", help="")
    parser.add_argument("--cam", type=str, default="upview", help="")
    parser.add_argument("--subtask", type=int, default=2, help="")
    args = parser.parse_args()

    gui = args.gui
    env_type = args.env
    cam_name = args.cam
    subtask = args.subtask

    main(gui=gui, env_type=env_type, cam_name=cam_name, subtask=subtask)
