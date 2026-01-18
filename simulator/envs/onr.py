import numpy as np
# from robosuite.models.arenas import EmptyArena
from robosuite.models.arenas.table_arena import TableArena

from simulator.objects import DoorObject, WallObject, SocketObject
from simulator.col_checker import ContactSensor
from util import geom
from .base import BaseEnv


MEAN_INIT_POS = {
    0: np.array([-0.5, 0., 0.743]),
    1: np.array([0.22, -0.1, 0.743]),
    2: np.array([0.06, -0.05, 0.743]),
}
STD_INIT_POS = {
    0: np.array([0.1, 0.2, 0.0]),
    1: np.array([0.01, 0.02, 0.0]),
    2: np.array([0.01, 0.02, 0.0]),
}
MEAN_INIT_YAW = {0: 0.0, 1: 0.05, 2: 0.0}
STD_INIT_YAW = {0: 0.2, 1: 0.07, 2: 0.0}
MEAN_INIT_HINGE = {0: 0.0, 1: 0.00, 2: 0.3}
STD_INIT_HINGE = {0: 0.0, 1: 0.00, 2: 0.025}


class ONRDoorEnv(BaseEnv):
    def reset(self, initial_pos=None, subtask=0, **kwargs):
        if initial_pos is None:
            self._init_robot_states = {
                "pos": np.random.normal(
                    MEAN_INIT_POS[subtask], STD_INIT_POS[subtask], size=3
                ),
                "yaw": np.random.normal(MEAN_INIT_YAW[subtask], STD_INIT_YAW[subtask]),
            }
        else:
            self._init_robot_states = initial_pos
        if subtask == 2:
            self._slide_joint_random = np.max(
                (
                    np.random.normal(MEAN_INIT_HINGE[subtask], STD_INIT_HINGE[subtask]),
                    0.08,
                )
            )
        else:
            self._slide_joint_random = 0

        out = super().reset(
            initial_pos=self._init_robot_states, subtask=subtask, **kwargs
        )

        self._success = False
        self._success_time = None
        return out

    def _load_model(self):
        # Create an environment
        super()._load_model()

        # Create an environment
        self.table_offset = np.array((0.3, 0.5, 0.75))
        self.table_size = np.array((0.5, 0.5, 0.05))
        self.arena = TableArena(table_full_size=self.table_size, table_offset=self.table_offset, has_legs=True)


        self.door = DoorObject(name="Door", friction=0.0, damping=0.1, type="onr")
        self.walls = {
            "wall{}".format(idx): WallObject(name="wall{}".format(idx))
            for idx in range(3)
        }
        self.socket = SocketObject(name="Socket")


        self.world.merge(self.arena)
        # self.world.merge_assets(self.shelf)
        # self.world.worldbody.append(self.shelf.get_obj())
        self.world.merge_assets(self.door)
        self.world.worldbody.append(self.door.get_obj())

        self.world.merge_assets(self.socket)
        self.world.worldbody.append(self.socket.get_obj())
        
        # print(self.door.joints)# = '0.7, 0, 1.0'

        for contact in self.door.contact:
            self.world.contact.append(contact)

        for wall in self.walls.values():
            self.world.merge_assets(wall)
            self.world.worldbody.append(wall.get_obj())

        for equality in self.door.equality:
            self.world.equality.append(equality)

    def _reset_objects(self):
        if "_slide_joint_random" in self.__dict__:
            for joint in self.door.joints:
                joint_id = self.sim.model.joint_name2id(joint)
                joint_qposadr = self.sim.model.jnt_qposadr[joint_id]
                if joint == "Door_slide":
                    self.sim.data.qpos[joint_qposadr] = self._slide_joint_random
                else:
                    self.sim.data.qpos[joint_qposadr] = 0

        # Initial Config
        door_body_id = self.sim.model.body_name2id(self.door.root_body)
        self.sim.model.body_pos[door_body_id] = np.array([0.7, 0, 0])

        wall_body_id = self.sim.model.body_name2id(self.walls["wall0"].root_body)
        self.sim.model.body_pos[wall_body_id] = np.array([0.7, -1.25, 0])

        wall_body_id = self.sim.model.body_name2id(self.walls["wall1"].root_body)
        self.sim.model.body_pos[wall_body_id] = np.array([0.7, 1.25, 0])

        wall_body_id = self.sim.model.body_name2id(self.walls["wall2"].root_body)
        self.sim.model.body_pos[wall_body_id] = np.array([0.7, 0.0, -1.4])

        for joint in self.socket.joints:
            joint_id = self.sim.model.joint_name2id(joint)
            joint_qposadr = self.sim.model.jnt_qposadr[joint_id]
            self.sim.data.qpos[joint_qposadr:joint_qposadr + 3] = [0.45, 0.3, 0.82]


    def _get_door_angle(self):
        joint = "Door_slide"
        joint_id = self.sim.model.joint_name2id(joint)
        joint_qposadr = self.sim.model.jnt_qposadr[joint_id]
        value = np.copy(self.sim.data.qpos[joint_qposadr])
        return value

    def _get_socket_pos(self):
        joint = "Socket_joint0"
        joint_id = self.sim.model.joint_name2id(joint)
        joint_qposadr = self.sim.model.jnt_qposadr[joint_id]
        value = np.copy(self.sim.data.qpos[joint_qposadr:joint_qposadr + 3])
        return value

    def _check_success(self):
        if self._subtask == 1:
            if self.door_angle > 0.043633:
                if self._success_time is None:
                    self._success_time = self.cur_time
        if self._subtask == 2:
            if self.controller.get_state() == 1:
                if self.robot_pos[0] > 0.5 and np.absolute(self.robot_pos[1]) < 0.3:
                    if self._success_time is None:
                        self._success_time = self.cur_time
                else:
                    self._success_time = None
        if self._subtask == 0:
            if self.controller.get_state() == 1:
                target = np.array([0.1, 0.0, 0.85])
                displacement = (target - self.robot_pos)[0:2]
                distance = np.linalg.norm(displacement)
                yaw = geom.quat_to_euler(self.robot_quat)[2]
                if np.absolute(yaw) < 0.3 and (distance < 0.1 or displacement[0] < 0):
                    if self._success_time is None:
                        self._success_time = self.cur_time
                else:
                    self._success_time = None
        if self._success_time is not None and self.cur_time - self._success_time > 0.5:
            self._success = True
            if self._subtask == 1 and self.cur_time - self._success_time < 1.0:
                self._success = False

        if self._subtask == 1 and self.cur_time > 20:
            self._success = True

    @property
    def door_angle(self):
        return self._get_door_angle()


    def _reset_robot(self, initial_pos, **kwargs):
        super()._reset_robot(initial_pos=initial_pos, **kwargs)
        self.contact_sensors = {
            'left_contact': ContactSensor(self.sim , robot={'left_gripper': self.grippers['left']}, objects={'door': self.door, **self.walls}),
            'right_contact': ContactSensor(self.sim , robot={'right_gripper': self.grippers['right']}, objects={'door': self.door, **self.walls}),
            'bimanual_contact_door': ContactSensor(self.sim , robot=self.grippers, objects={'door': self.door}),
            }

    def _update_obs(self):
        super()._update_obs()
        self._cur_obs["contact_sensor"] = self._read_contact_sensors()

    def _read_contact_sensors(self):        
        return {key: sensor.get_state() for key, sensor in self.contact_sensors.items()}
