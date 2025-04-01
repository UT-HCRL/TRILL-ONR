import numpy as np
from robosuite.models.arenas import EmptyArena
from .base import BaseEnv
from xml.etree.ElementTree import Element, SubElement


class EmptyEnv(BaseEnv):
    def _load_model(self):
        # Create an environment
        super()._load_model()

        # Add a ground plane
        ground = Element('geom')
        ground.set('name', 'ground')
        ground.set('type', 'plane')
        ground.set('size', '10 10 0.01')
        ground.set('pos', '0 0 0')
        ground.set('rgba', '0.5 0.5 0.5 1')
        self.world.worldbody.append(ground)

        # Add lighting
        light = Element('light')
        light.set('pos', '1 1 1.5')
        light.set('dir', '-0.19245 -0.19245 -0.96225')
        light.set('directional', 'true')
        light.set('castshadow', 'false')
        self.world.worldbody.append(light)

    def _reset_objects(self):
        # No objects to reset in this simple environment
        pass

    def _check_success(self):
        # No success condition for this environment
        self._success = False
