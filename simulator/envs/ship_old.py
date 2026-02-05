import numpy as np
import os
from robosuite.models.arenas import EmptyArena
from .base import BaseEnv
from xml.etree.ElementTree import Element, SubElement

class ShipEnv(BaseEnv):
    def _load_model(self):
        # Initialize base environment
        super()._load_model()
        workspace_path = os.getcwd()

        # ============== DEFAULT CLASSES SETUP ==============
        self._setup_default_classes()

        # ============== ASSETS SETUP ==============
        self._setup_assets(workspace_path)

        # ============== FLOOR SETUP ==============
        self._setup_floor()

        # ============== OBJECTS SETUP ==============
        self._setup_objects()

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
        visual_class.append(visual_geom)
        default.append(visual_class)

        # Collision class for physical elements
        collision_class = Element('default')
        collision_class.set('class', 'collision')
        collision_geom = Element('geom')
        collision_geom.set('group', '3')
        collision_geom.set('type', 'mesh')
        collision_class.append(collision_geom)
        default.append(collision_class)

        self.world.root.append(default)

    def _setup_assets(self, workspace_path):
        """Setup all assets including materials, textures, and meshes"""
        asset = Element('asset')
        
        # ===== Materials =====
        materials = {
            'metal': {
                'specular': '0.9',
                'shininess': '0.8',
                'rgba': '0.8 0.8 0.85 1.0'
            },
            'wood': {
                'specular': '0.3',
                'shininess': '0.2',
                'texture': 'light-wood',
                'texrepeat': '8 8',
                'rgba': '0.95 0.9 0.8 1.0'
            },
            'Engine': {
                'specular': '0.5',
                'shininess': '0.2',
                'texture': 'engine-texture',
                'texrepeat': '1 1',
                'rgba': '1.0 1.0 1.0 1.0'
            }
        }

        for name, props in materials.items():
            material = Element('material')
            material.set('name', name)
            for key, value in props.items():
                material.set(key, value)
            asset.append(material)

        # ===== Textures =====
        # Wood texture
        texture = Element('texture')
        texture.set('name', 'light-wood')
        texture.set('type', '2d')
        texture.set('file', os.path.join(workspace_path, 'models/textures/light-wood.png'))
        asset.append(texture)

        # Engine textures
        engine_texture = Element('texture')
        engine_texture.set('name', 'engine-texture')
        engine_texture.set('type', '2d')
        engine_texture.set('file', os.path.join(workspace_path, 'custom_env/ShipScene/ShipScene/engine_basecolor.png'))
        asset.append(engine_texture)

        engine_normal_texture = Element('texture')
        engine_normal_texture.set('name', 'engine-normal')
        engine_normal_texture.set('type', '2d')
        engine_normal_texture.set('file', '/home/sanjayovs/Desktop/1_Projects/HARMONIC/Unity Simulations/Unity Projects/LEIA_BT_SIM_V0/Assets/Engine room devices/Textures/engine_normal.png')
        asset.append(engine_normal_texture)

        # ===== Meshes =====
        # Engine mesh
        engine_mesh = Element('mesh')
        engine_mesh.set('file', os.path.join(workspace_path, 'custom_env/ShipScene/ShipScene/objs/Engine.obj'))
        engine_mesh.set('name', 'engine')
        engine_mesh.set('scale', '0.35 0.3 0.35')
        asset.append(engine_mesh)

        self.world.root.append(asset)

    def _setup_floor(self):
        """Setup the wooden floor and metal walls"""
        # Floor dimensions
        floor_size = 5
        wall_height = 1  # Reduced from 2 to 1
        wall_thickness = 0.1

        # ===== Floor =====
        ground = Element('geom')
        ground.set('name', 'wooden_ground')
        ground.set('type', 'box')
        ground.set('size', f'{floor_size} {floor_size} 0.01')
        ground.set('pos', '0 0 0')
        ground.set('material', 'wood')
        ground.set('rgba', '0.8 0.7 0.6 1.0')
        self.world.worldbody.append(ground)

        # ===== Walls =====
        # Back wall
        back_wall = Element('geom')
        back_wall.set('name', 'back_wall')
        back_wall.set('type', 'box')
        back_wall.set('size', f'{floor_size} {wall_thickness} {wall_height}')
        back_wall.set('pos', f'0 {floor_size} {wall_height/2}')
        back_wall.set('material', 'metal')
        back_wall.set('rgba', '0.8 0.8 0.85 1.0')
        self.world.worldbody.append(back_wall)

        # Front wall
        front_wall = Element('geom')
        front_wall.set('name', 'front_wall')
        front_wall.set('type', 'box')
        front_wall.set('size', f'{floor_size} {wall_thickness} {wall_height}')
        front_wall.set('pos', f'0 {-floor_size} {wall_height/2}')
        front_wall.set('material', 'metal')
        front_wall.set('rgba', '0.8 0.8 0.85 1.0')
        self.world.worldbody.append(front_wall)

        # Left wall
        left_wall = Element('geom')
        left_wall.set('name', 'left_wall')
        left_wall.set('type', 'box')
        left_wall.set('size', f'{wall_thickness} {floor_size} {wall_height}')
        left_wall.set('pos', f'{-floor_size} 0 {wall_height/2}')
        left_wall.set('material', 'metal')
        left_wall.set('rgba', '0.8 0.8 0.85 1.0')
        self.world.worldbody.append(left_wall)

        # Right wall
        right_wall = Element('geom')
        right_wall.set('name', 'right_wall')
        right_wall.set('type', 'box')
        right_wall.set('size', f'{wall_thickness} {floor_size} {wall_height}')
        right_wall.set('pos', f'{floor_size} 0 {wall_height/2}')
        right_wall.set('material', 'metal')
        right_wall.set('rgba', '0.8 0.8 0.85 1.0')
        self.world.worldbody.append(right_wall)

    def _setup_objects(self):
        """Setup all objects in the scene"""
        # Get the workspace path for mesh files
        workspace_path = os.getcwd()
        engine_mesh_path = os.path.join(workspace_path, 'custom_env/ShipScene/ShipScene/objs/Engine.obj')
        
        # Engine dimensions and spacing
        engine_scale = (0.3, 0.25, 0.3)
        engine_spacing = 1.75  # Space between engines
        base_y = 4  # Base Y position
        
        # First Engine
        self._add_new_object(
            name='engine_1',
            mesh_path=engine_mesh_path,
            material='Engine',
            position=(-2.5, base_y, 0),
            rotation=(1.57, 1.57, 0),
            scale=engine_scale
        )
        
        # Second Engine
        self._add_new_object(
            name='engine_2',
            mesh_path=engine_mesh_path,
            material='Engine',
            position=(-2.5, base_y - engine_spacing, 0),
            rotation=(1.57, 1.57, 0),
            scale=engine_scale
        )
        


    def _setup_lighting(self):
        """Setup scene lighting"""
        light = Element('light')
        light.set('pos', '1 1 1.5')
        light.set('dir', '-0.19245 -0.19245 -0.96225')
        light.set('directional', 'true')
        light.set('castshadow', 'false')
        self.world.worldbody.append(light)

    def _add_new_object(self, name, mesh_path, material, position, rotation, scale):
        """Helper method to add new objects to the scene
        Args:
            name (str): Name of the object
            mesh_path (str): Path to the mesh file
            material (str): Name of the material to apply
            position (tuple): (x, y, z) position
            rotation (tuple): (rx, ry, rz) rotation in radians
            scale (tuple): (sx, sy, sz) scale factors
        """
        # Add mesh to assets
        mesh = Element('mesh')
        mesh.set('file', mesh_path)
        mesh.set('name', name)
        mesh.set('scale', f"{scale[0]} {scale[1]} {scale[2]}")
        self.world.root.find('asset').append(mesh)

        # Add object body
        body = Element('body')
        body.set('name', name)
        body.set('pos', f"{position[0]} {position[1]} {position[2]}")
        body.set('euler', f"{rotation[0]} {rotation[1]} {rotation[2]}")
        
        # Add visual geom (for rendering)
        visual_geom = Element('geom')
        visual_geom.set('mesh', name)
        visual_geom.set('material', material)
        visual_geom.set('class', 'visual')
        visual_geom.set('group', '2')
        body.append(visual_geom)
        
        # Add collision geom (for physics)
        collision_geom = Element('geom')
        collision_geom.set('mesh', name)
        collision_geom.set('class', 'collision')
        collision_geom.set('group', '3')
        body.append(collision_geom)
        
        self.world.worldbody.append(body)

    def _reset_robot(self, initial_pos=None):
        """Override base class to set custom robot position"""
        # If no initial position provided, use our custom position
        if initial_pos is None:
            # Convert list to dictionary format expected by controller
            initial_pos = {
                "pos": [-3, 0.5, 0.743],  # x, y, z position
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