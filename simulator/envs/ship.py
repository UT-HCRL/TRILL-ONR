import numpy as np
import os
from robosuite.models.arenas import EmptyArena

import util.geom
from .base import BaseEnv
from xml.etree.ElementTree import Element, SubElement
from simulator.objects import DoorObject, TrayObject
from simulator.objects import TableObject

class ShipEnv(BaseEnv):
    def __init__(self):
        self.scene_scale = 0.022
        super().__init__()

    def _load_model(self):
        super()._load_model()
        workspace_path = os.getcwd()
        
        self._setup_default_classes()
        self._setup_assets(workspace_path)
        self._setup_scene()
        self._setup_lighting()
        self._setup_door()
        self._setup_table()
        # self._setup_tray()

    def _setup_default_classes(self):
        default = Element('default')
        
        # Visual class
        visual_class = Element('default')
        visual_class.set('class', 'visual')
        visual_geom = Element('geom')
        visual_geom.set('group', '2')
        visual_geom.set('type', 'mesh')
        visual_geom.set('contype', '0')
        visual_geom.set('conaffinity', '0')
        visual_geom.set('solimp', '0.6 0.7 0.001')
        visual_geom.set('solref', '0.005 1')
        visual_geom.set('margin', '0.001')
        visual_class.append(visual_geom)
        default.append(visual_class)

        # Collision class
        collision_class = Element('default')
        collision_class.set('class', 'collision')
        collision_geom = Element('geom')
        collision_geom.set('group', '3')
        collision_geom.set('type', 'mesh')
        collision_geom.set('contype', '1')
        collision_geom.set('conaffinity', '1')
        collision_geom.set('solimp', '0.6 0.7 0.001')
        collision_geom.set('solref', '0.005 1')
        collision_geom.set('margin', '0.001')
        collision_geom.set('friction', '0.5 0.1 0.1')
        collision_class.append(collision_geom)
        default.append(collision_class)

        self.world.root.append(default)

    def _setup_assets(self, workspace_path):
        asset = Element('asset')
        
        # Materials configuration
        materials = {
            'Box_Material': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.905660 0.276824 0.276824 1.000000'},
            'Device1': {'texture': 'device1_basecolor', 'specular': '0.5', 'shininess': '0.2'},
            'Device4': {'texture': 'device4_basecolor', 'specular': '0.5', 'shininess': '0.2'},
            'Engine': {'texture': 'engine_basecolor', 'specular': '0.5', 'shininess': '0.2'},
            'Floor1': {'specular': '0.5', 'shininess': '0.2', 'rgba': '1.000000 0.939428 0.636792 1.000000'},
            'M_Room_01': {'texture': 'T_Room_01_BC 1', 'specular': '0.5', 'shininess': '0.2'},
            'Oxygen': {'texture': 'oxygen_basecolor', 'specular': '0.5', 'shininess': '0.2'},
            'ShipControl': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.858491 0.858491 0.858491 1.000000'},
            'UpperDeck_Walls': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.811321 0.811321 0.811321 1.000000'},
            'dark': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.120154 0.120154 0.120154 1.000000'},
            'floor': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.255981 0.255981 0.255981 1.000000'},
            'plain': {'specular': '0.5', 'shininess': '0.2', 'rgba': '1.000000 1.000000 1.000000 1.000000'}
        }

        # Textures configuration
        textures = {
            'device1_basecolor': 'device1_basecolor.png',
            'device4_basecolor': 'device4_basecolor.png',
            'engine_basecolor': 'engine_basecolor.png',
            'T_Room_01_BC 1': 'T_Room_01_BC 1.png',
            'oxygen_basecolor': 'oxygen_basecolor.png'
        }

        # Add textures
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

        # Add meshes
        for i in range(12):
            mesh_name = f'ShipScene_{i}'
            mesh = Element('mesh')
            mesh.set('file', os.path.join(workspace_path, f'custom_env/ShipScene/ShipScene/{mesh_name}.obj'))
            mesh.set('name', mesh_name)
            mesh.set('scale', f'{self.scene_scale} {self.scene_scale} {self.scene_scale}')
            asset.append(mesh)

        self.world.root.append(asset)

    def _setup_scene(self):
        ship_body = Element('body')
        ship_body.set('name', 'ShipScene')
        ship_body.set('pos', '5 5 -2.374')
        ship_body.set('euler', '1.57 3.14 0')
        
        # Add custom wall
        wall_body = Element('body')
        wall_body.set('name', 'SimpleWall')
        wall_body.set('pos', '7.2 2.73 -7.19')
        wall_body.set('euler', '0 1.57 0')
        
        wall_geom = Element('geom')
        wall_geom.set('type', 'box')
        wall_geom.set('size', '0.075 1.8 1.0')
        wall_geom.set('material', 'UpperDeck_Walls')
        wall_geom.set('contype', '1')
        wall_geom.set('conaffinity', '1')
        wall_geom.set('friction', '0.5 0.1 0.1')
        wall_body.append(wall_geom)
        ship_body.append(wall_body)
        
        # Floor collision properties
        floor_properties = {
            'contype': '1',
            'conaffinity': '1',
            'solimp': '0.6 0.7 0.001',
            'solref': '0.005 1',
            'friction': '1.0 0.5 0.1',
            'margin': '0.001',
            'class': 'collision'
        }
        
        # Scene meshes configuration
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
            geom = Element('geom')
            geom.set('mesh', mesh_name)
            geom.set('material', material)
            geom.set('group', '2')
            
            if collision_type == 'floor':
                for key, value in floor_properties.items():
                    geom.set(key, value)
            else:
                geom.set('contype', '0')
                geom.set('conaffinity', '0')
                geom.set('class', 'visual')
            
            ship_body.append(geom)

        self.world.worldbody.append(ship_body)

    def _setup_lighting(self):
        # Main directional light from above
        main_light = Element('light')
        main_light.set('name', 'main')
        main_light.set('pos', '1 1 3.0')
        main_light.set('dir', '-0.19245 -0.19245 -0.96225')
        main_light.set('directional', 'true')
        main_light.set('castshadow', 'false')
        main_light.set('diffuse', '0.7 0.7 0.7')
        main_light.set('specular', '0.5 0.5 0.5')
        self.world.worldbody.append(main_light)

        # Fill light from the opposite side
        fill_light = Element('light')
        fill_light.set('name', 'fill')
        fill_light.set('pos', '-2 -2 2.5')
        fill_light.set('dir', '0.19245 0.19245 -0.96225')
        fill_light.set('directional', 'true')
        fill_light.set('castshadow', 'false')
        fill_light.set('diffuse', '0.4 0.4 0.4')
        fill_light.set('specular', '0.1 0.1 0.1')
        self.world.worldbody.append(fill_light)

        # Ambient light for overall scene brightness
        ambient_light = Element('light')
        ambient_light.set('name', 'ambient')
        ambient_light.set('pos', '0 0 4.0')
        ambient_light.set('dir', '0 0 -1')
        ambient_light.set('directional', 'true')
        ambient_light.set('castshadow', 'false')
        ambient_light.set('diffuse', '0.3 0.3 0.3')
        ambient_light.set('specular', '0.1 0.1 0.1')
        self.world.worldbody.append(ambient_light)

    def _setup_door(self):
        self.door = DoorObject(
            name="ShipDoor",
            friction=0.0,
            damping=0.1,
            type="hinge"
        )

        self.world.merge_assets(self.door)
        self.world.worldbody.append(self.door.get_obj())

        for contact in self.door.contact:
            self.world.contact.append(contact)

        for equality in self.door.equality:
            self.world.equality.append(equality)

    def _setup_table(self):
        self.table = TableObject(name="ShipTable")

        self.world.merge_assets(self.table)
        self.world.worldbody.append(self.table.get_obj())

        for contact in self.table.contact:
            self.world.contact.append(contact)

        for equality in self.table.equality:
            self.world.equality.append(equality)

    # def _setup_tray(self):
    #     self.tray = TrayObject(name="ShipTray")
    #
    #     self.world.merge_assets(self.tray)
    #     self.world.worldbody.append(self.tray.get_obj())
    #
    #     for contact in self.tray.contact:
    #         self.world.contact.append(contact)
    #
    #     for equality in self.tray.equality:
    #         self.world.equality.append(equality)

    def _reset_robot(self, initial_pos=None):
        if initial_pos is None:
            initial_pos = {
                "pos": [0.1, 2.75, 0.741],
                "yaw": -1.5708
            }
        elif isinstance(initial_pos, list):
            initial_pos = {
                "pos": initial_pos[:3],
                "yaw": initial_pos[5]
            }
        
        super()._reset_robot(initial_pos=initial_pos)

    def _reset_objects(self):
        if hasattr(self, 'door'):
            door_body_id = self.sim.model.body_name2id(self.door.root_body)
            self.sim.model.body_pos[door_body_id] = np.array([-3.7, -2.15, 0.2])
            self.sim.model.body_quat[door_body_id] = np.array([0.7071, 0.0, 0.0, 0.7071])
            
            for joint in self.door.joints:
                joint_id = self.sim.model.joint_name2id(joint)
                joint_qposadr = self.sim.model.jnt_qposadr[joint_id]
                self.sim.data.qpos[joint_qposadr] = 0.0

        if hasattr(self, 'table'):
            table_body_id = self.sim.model.body_name2id(self.table.root_body)
            self.sim.model.body_pos[table_body_id] = np.array([0., 0., 0.0])
            self.sim.model.body_quat[table_body_id] = np.array([0.7071, 0.0, 0.0, 0.7071])

            for joint in self.table.joints:
                joint_id = self.sim.model.joint_name2id(joint)
                joint_qposadr = self.sim.model.jnt_qposadr[joint_id]
                self.sim.data.qpos[joint_qposadr:joint_qposadr+3] = np.array([0.0, 1.0, 0.0])

        # if hasattr(self, 'table') and hasattr(self, 'tray'):
        #     tray_body_id = self.sim.model.body_name2id(self.tray.root_body)
        #     self.sim.model.body_pos[tray_body_id] = np.array([0., 0., 0])
        #     self.sim.model.body_quat[tray_body_id] = np.array([0.7071, 0.0, 0.0, 0.7071])
        #
        #     for joint in self.tray.joints:
        #         joint_id = self.sim.model.joint_name2id(joint)
        #         joint_qposadr = self.sim.model.jnt_qposadr[joint_id]
        #         self.sim.data.qpos[joint_qposadr:joint_qposadr+3] = np.array([0.0, 1.1, 0.65])
        #         self.sim.data.qpos[joint_qposadr+3:joint_qposadr+7] = util.geom.euler_to_quat([np.pi/2, 0., 0.])

    def _check_success(self):
        self._success = False 

    # Robot movement methods
    # def walk_forward(self):
    #     self.controller.update_trajectory({}, locomotion="walk_forward")

    # def walk_backward(self):
    #     self.controller.update_trajectory({}, locomotion="walk_backward")

    # def strafe_left(self):
    #     self.controller.update_trajectory({}, locomotion="sidewalk_left")

    # def strafe_right(self):
    #     self.controller.update_trajectory({}, locomotion="sidewalk_right")

    # def turn_left(self):
    #     self.controller.update_trajectory({}, locomotion="turning_left")

    # def turn_right(self):
    #     self.controller.update_trajectory({}, locomotion="turning_right")

    # def balance(self):
    #     self.controller.update_trajectory({}, locomotion="balance")
    #     self.controller.update_trajectory({}, locomotion="balance")

    # Collision visibility methods
    def show_all_colliders(self):
        self.sim.model.vis.flags[3:6] = 1
        self.sim.forward()
        return True

    def toggle_collision_visibility(self):
        current_state = self.sim.model.vis.flags[3]
        new_state = 1 - current_state
        self.sim.model.vis.flags[3:6] = new_state
        self.sim.forward()
        return new_state == 1

    # Door-related methods
    def _get_door_angle(self):
        if hasattr(self, 'door'):
            joint = "ShipDoor_hinge"
            joint_id = self.sim.model.joint_name2id(joint)
            joint_qposadr = self.sim.model.jnt_qposadr[joint_id]
            return np.copy(self.sim.data.qpos[joint_qposadr])
        return 0.0

    @property
    def door_angle(self):
        return self._get_door_angle() 