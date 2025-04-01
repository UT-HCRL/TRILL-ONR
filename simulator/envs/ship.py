import numpy as np
import os
from robosuite.models.arenas import EmptyArena
from .base import BaseEnv
from xml.etree.ElementTree import Element, SubElement

class ShipEnv(BaseEnv):
    def __init__(self):
        self.scene_scale = 0.022  # Reduced scale to better fit the robot
        super().__init__()

    def _load_model(self):
        # Initialize base environment
        super()._load_model()
        workspace_path = os.getcwd()

        # ============== DEFAULT CLASSES SETUP ==============
        self._setup_default_classes()

        # ============== ASSETS SETUP ==============
        self._setup_assets(workspace_path)

        # ============== SCENE SETUP ==============
        self._setup_scene()

        # ============== LIGHTING SETUP ==============
        self._setup_lighting()

    def _setup_default_classes(self):
        """Setup default classes for visual and collision properties"""
        default = Element('default')
        
        # Visual class for non-colliding elements
        visual_class = Element('default')
        visual_class.set('class', 'visual')
        visual_geom = Element('geom')
        visual_geom.set('group', '2')
        visual_geom.set('type', 'mesh')
        visual_geom.set('contype', '0')
        visual_geom.set('conaffinity', '0')
        visual_geom.set('solimp', '0.6 0.7 0.001')  # More conservative values
        visual_geom.set('solref', '0.005 1')  # Reduced damping
        visual_geom.set('margin', '0.001')  # Small margin
        visual_class.append(visual_geom)
        default.append(visual_class)

        # Collision class for physical elements
        collision_class = Element('default')
        collision_class.set('class', 'collision')
        collision_geom = Element('geom')
        collision_geom.set('group', '3')
        collision_geom.set('type', 'mesh')
        collision_geom.set('contype', '1')
        collision_geom.set('conaffinity', '1')
        collision_geom.set('solimp', '0.6 0.7 0.001')  # More conservative values
        collision_geom.set('solref', '0.005 1')  # Reduced damping
        collision_geom.set('margin', '0.001')  # Small margin
        collision_geom.set('friction', '0.5 0.1 0.1')  # Moderate friction
        collision_class.append(collision_geom)
        default.append(collision_class)

        self.world.root.append(default)

    def _setup_assets(self, workspace_path):
        """Setup all assets including materials and textures"""
        asset = Element('asset')
        
        # ===== Materials =====
        materials = {
            'Box_Material': {
                'specular': '0.5',
                'shininess': '0.2',
                'rgba': '0.905660 0.276824 0.276824 1.000000'
            },
            'Device1': {
                'texture': 'device1_basecolor',
                'specular': '0.5',
                'shininess': '0.2'
            },
            'Device4': {
                'texture': 'device4_basecolor',
                'specular': '0.5',
                'shininess': '0.2'
            },
            'Engine': {
                'texture': 'engine_basecolor',
                'specular': '0.5',
                'shininess': '0.2'
            },
            'Floor1': {
                'specular': '0.5',
                'shininess': '0.2',
                'rgba': '1.000000 0.939428 0.636792 1.000000'
            },
            'M_Room_01': {
                'texture': 'T_Room_01_BC 1',
                'specular': '0.5',
                'shininess': '0.2'
            },
            'Oxygen': {
                'texture': 'oxygen_basecolor',
                'specular': '0.5',
                'shininess': '0.2'
            },
            'ShipControl': {
                'specular': '0.5',
                'shininess': '0.2',
                'rgba': '0.858491 0.858491 0.858491 1.000000'
            },
            'UpperDeck_Walls': {
                'specular': '0.5',
                'shininess': '0.2',
                'rgba': '0.811321 0.811321 0.811321 1.000000'
            },
            'dark': {
                'specular': '0.5',
                'shininess': '0.2',
                'rgba': '0.120154 0.120154 0.120154 1.000000'
            },
            'floor': {
                'specular': '0.5',
                'shininess': '0.2',
                'rgba': '0.255981 0.255981 0.255981 1.000000'
            },
            'plain': {
                'specular': '0.5',
                'shininess': '0.2',
                'rgba': '1.000000 1.000000 1.000000 1.000000'
            }
        }

        # Add textures
        textures = {
            'device1_basecolor': 'device1_basecolor.png',
            'device4_basecolor': 'device4_basecolor.png',
            'engine_basecolor': 'engine_basecolor.png',
            'T_Room_01_BC 1': 'T_Room_01_BC 1.png',
            'oxygen_basecolor': 'oxygen_basecolor.png'
        }

        for name, file in textures.items():
            texture = Element('texture')
            texture.set('type', '2d')
            texture.set('name', name)
            texture.set('file', os.path.join(workspace_path, 'custom_env/ShipScene/ShipScene', file))
            asset.append(texture)

        # Add materials
        for name, props in materials.items():
            material = Element('material')
            material.set('name', name)
            for key, value in props.items():
                material.set(key, value)
            asset.append(material)

        # ===== Meshes =====
        # Add visual meshes with scene scale
        for i in range(12):
            mesh_name = f'ShipScene_{i}'
            mesh = Element('mesh')
            mesh.set('file', os.path.join(workspace_path, f'custom_env/ShipScene/ShipScene/{mesh_name}.obj'))
            mesh.set('name', mesh_name)
            mesh.set('scale', f'{self.scene_scale} {self.scene_scale} {self.scene_scale}')  # Use scene scale for all meshes
            asset.append(mesh)

        self.world.root.append(asset)

    def _setup_scene(self):
        """Setup the custom scene with multiple objects"""
        # Create a parent body for the entire ship
        ship_body = Element('body')
        ship_body.set('name', 'ShipScene')
        
        # Set the position and rotation of the entire scene
        # Position: (x, y, z) - negative x moves left, positive y moves right, positive z moves up
        # Euler angles: (roll, pitch, yaw) in radians
        # Position the scene so the robot is inside the ship
        ship_body.set('pos', '5 5 -2.374')  # Restore previous position
        ship_body.set('euler', '1.57 3.14 0')  # Restore previous rotation
        
        # Define collision properties for floor only
        floor_properties = {
            'contype': '1',  # Enable collisions
            'conaffinity': '1',  # Enable collisions
            'solimp': '0.6 0.7 0.001',  # Conservative values for stability
            'solref': '0.005 1',  # Reduced damping
            'friction': '0.5 0.1 0.1',  # Moderate friction
            'margin': '0.001',  # Small margin for better contact
            'class': 'collision'  # Use collision class
        }
        
        # Add visual geometries with specific collision properties
        visual_meshes = [
            ('ShipScene_0', 'floor', 'floor'),
            ('ShipScene_1', 'M_Room_01', 'walls'),
            ('ShipScene_2', 'ShipControl', 'devices'),
            ('ShipScene_3', 'dark', 'walls'),
            ('ShipScene_4', 'plain', 'walls'),
            ('ShipScene_5', 'Engine', 'engine'),
            ('ShipScene_6', 'Oxygen', 'devices'),
            ('ShipScene_7', 'Floor1', 'floor'),
            ('ShipScene_8', 'Box_Material', 'walls'),
            ('ShipScene_9', 'Device4', 'devices'),
            ('ShipScene_10', 'Device1', 'devices'),
            ('ShipScene_11', 'UpperDeck_Walls', 'walls')
        ]

        for mesh_name, material, collision_type in visual_meshes:
            # Add geometry that serves as both visual and collision
            geom = Element('geom')
            geom.set('mesh', mesh_name)
            geom.set('material', material)
            geom.set('group', '2')  # Visual group
            
            # Only enable collisions for floor objects
            if collision_type == 'floor':
                for key, value in floor_properties.items():
                    geom.set(key, value)
            else:
                # Disable collisions for all other objects
                geom.set('contype', '0')
                geom.set('conaffinity', '0')
                geom.set('class', 'visual')
            
            ship_body.append(geom)

        self.world.worldbody.append(ship_body)

    def _setup_lighting(self):
        """Setup scene lighting"""
        light = Element('light')
        light.set('pos', '1 1 1.5')
        light.set('dir', '-0.19245 -0.19245 -0.96225')
        light.set('directional', 'true')
        light.set('castshadow', 'false')
        self.world.worldbody.append(light)

    def _reset_robot(self, initial_pos=None):
        """Override base class to set custom robot position"""
        # If no initial position provided, use our custom position
        if initial_pos is None:
            # Position the robot inside the ship
            # x: -2.5 (to match scene position), y: 0 (center), z: 0.743 (slightly above floor)
            initial_pos = {
                "pos": [-2.5, 0, 0.742],  # Position relative to the scene
                "quat": [1.0, 0.0, 0.0, 0.0]  # quaternion for rotation (no rotation)
            }
        elif isinstance(initial_pos, list):
            # Convert list format [x, y, z, roll, pitch, yaw] to dictionary format
            pos = initial_pos[:3]
            # Convert Euler angles to quaternion (simplified for now)
            roll, pitch, yaw = initial_pos[3:]
            # For now, just use identity quaternion
            quat = [1.0, 0.0, 0.0, 0.0]
            initial_pos = {
                "pos": pos,
                "quat": quat
            }
        
        # Call parent class method with our position
        super()._reset_robot(initial_pos=initial_pos)

    def _reset_objects(self):
        """Reset object positions and states"""
        pass

    def _check_success(self):
        """Check if the task is complete"""
        self._success = False 

    def walk_forward(self):
        """Make the robot walk forward"""
        self.controller.update_trajectory({}, locomotion="walk_forward")

    def walk_backward(self):
        """Make the robot walk backward"""
        self.controller.update_trajectory({}, locomotion="walk_backward")

    def strafe_left(self):
        """Make the robot strafe to the left"""
        self.controller.update_trajectory({}, locomotion="sidewalk_left")

    def strafe_right(self):
        """Make the robot strafe to the right"""
        self.controller.update_trajectory({}, locomotion="sidewalk_right")

    def turn_left(self):
        """Make the robot turn left"""
        self.controller.update_trajectory({}, locomotion="turning_left")

    def turn_right(self):
        """Make the robot turn right"""
        self.controller.update_trajectory({}, locomotion="turning_right")

    def balance(self):
        """Make the robot balance in place"""
        self.controller.update_trajectory({}, locomotion="balance") 
        self.controller.update_trajectory({}, locomotion="balance") 

    def show_all_colliders(self):
        """Make all colliders visible in the simulation"""
        # Set visibility flags for all collision groups (group 3)
        self.sim.model.vis.flags[3] = 1  # Show collision meshes
        self.sim.model.vis.flags[4] = 1  # Show collision points
        self.sim.model.vis.flags[5] = 1  # Show collision lines
        
        # Update the visualization
        self.sim.forward()
        
        return True

    def toggle_collision_visibility(self):
        """Toggle the visibility of collision meshes in the simulation"""
        # Get the current visibility state of collision meshes (group 3)
        current_state = self.sim.model.vis.flags[3]
        
        # Toggle the state (0 = hidden, 1 = visible)
        self.sim.model.vis.flags[3] = 1 - current_state
        self.sim.model.vis.flags[4] = 1 - current_state  # Also toggle collision points
        self.sim.model.vis.flags[5] = 1 - current_state  # Also toggle collision lines
        
        # Update the visualization
        self.sim.forward()
        
        return self.sim.model.vis.flags[3] == 1  # Return True if now visible, False if hidden 