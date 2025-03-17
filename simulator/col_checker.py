from . import sim_util

class CollisionChecker(object):

    def __init__(self, sim, robot, objects, **kwargs) -> None:
        self._sim = sim
        self._robot = robot
        self._objects = objects
        self.reset()

    def reset(self):

        self._geom_info = {
            'robot': {},
            'object': {},
        }

        for key, robot in self._robot.items():
            self._geom_info['robot'].update(sim_util.get_col_geom(self._sim, robot))

        for key, object in self._objects.items():
            self._geom_info['object'].update(sim_util.get_col_geom(self._sim, object))


    def get_state(self):

        robot_object_col_state, robot_object_col_info = sim_util.check_col_state(self._sim, self._geom_info['robot'], self._geom_info['object'])
        self_col_state, self_col_info = sim_util.check_col_state(self._sim, self._geom_info['robot'], self._geom_info['robot'])

        col_state = robot_object_col_state or self_col_state
        col_info = robot_object_col_info + self_col_info

        return col_state, col_info
    

class ContactSensor:

    def __init__(self, sim, robot, objects, **kwargs):
        self.col_checker = CollisionChecker(sim, robot, objects)
        self._col_state = False

    def reset(self, **kwargs):
        self.col_checker.reset()
        self._col_state = False

    def get_state(self):
        col_state, col_info = self.col_checker.get_state()
        self._col_state = col_state
        return self.state

    @property
    def state(self):
        return self._col_state