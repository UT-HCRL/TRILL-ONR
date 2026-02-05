import numpy as np
from robosuite.models.arenas import EmptyArena
from .base import BaseEnv
from xml.etree.ElementTree import Element, SubElement

class CheckeredEnv(BaseEnv):
    def _load_model(self):
        # Create an environment
        super()._load_model()

        # Create checkered floor pattern
        size = 0.5  # Size of each square
        for i in range(-3, 4):
            for j in range(-3, 4):
                # Alternate colors based on position
                if (i + j) % 2 == 0:
                    rgba = [0.2, 0.2, 0.2, 1]  # Dark squares
                else:
                    rgba = [0.8, 0.8, 0.8, 1]  # Light squares
                
                # Create square geom as XML element
                geom = Element('geom')
                geom.set('name', f'square_{i}_{j}')
                geom.set('type', 'box')
                geom.set('size', f'{size} {size} 0.01')
                geom.set('pos', f'{i*size*2} {j*size*2} 0')
                geom.set('rgba', f'{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}')
                
                self.world.worldbody.append(geom)

        # Add a ground plane to prevent objects from falling through
        ground = Element('geom')
        ground.set('name', 'checkered_ground')
        ground.set('type', 'plane')
        ground.set('size', '10 10 0.01')
        ground.set('pos', '0 0 0')
        ground.set('rgba', '0.5 0.5 0.5 1')
        self.world.worldbody.append(ground)

        # Add lighting from base environment
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