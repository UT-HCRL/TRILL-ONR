import numpy as np
import os
from robosuite.models.arenas import EmptyArena
from .base import BaseEnv
from xml.etree.ElementTree import Element, SubElement

class ShipEnv2(BaseEnv):
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
        
        # Materials configuration from Ship2.xml
        materials = {
            'Base_metal': {'texture': 'Base_metal', 'specular': '0.5', 'shininess': '0.2'},
            'Brass-Matte1': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.820755 0.676179 0.000000 1.000000'},
            'Brass_-_Matte': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.952941 0.796078 0.486275 1.000000'},
            'Compressor': {'texture': 'compressor_basecolor', 'specular': '0.5', 'shininess': '0.2'},
            'ControlboardDetailed': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.4 0.4 0.45 1.0'},
            'DeckDetailed': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.6 0.6 0.65 1.0'},
            'Device1': {'texture': 'device1_basecolor', 'specular': '0.5', 'shininess': '0.2'},
            'Device4': {'texture': 'device4_basecolor', 'specular': '0.5', 'shininess': '0.2'},
            'Engine': {'texture': 'engine_basecolor', 'specular': '0.5', 'shininess': '0.2'},
            'LowerDeck_Walls': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.811321 0.811321 0.811321 1.000000'},
            'M_Room_01': {'texture': 'T_Room_01_BC 1', 'specular': '0.5', 'shininess': '0.2'},
            'Oxygen': {'texture': 'oxygen_basecolor', 'specular': '0.5', 'shininess': '0.2'},
            'ShipControl': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.858491 0.858491 0.858491 1.000000'},
            'Stainless_Steel_-_Polished': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.796078 0.796078 0.796078 1.000000'},
            'ThermostatBody1': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.311321 0.311321 0.311321 1.000000'},
            'ThermostatBody_4': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.000000 0.367925 0.056851 1.000000'},
            'Thermostat_Body_2': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.556604 0.000000 0.000000 1.000000'},
            'Thermostat_Body_3': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.000000 0.174627 0.943396 1.000000'},
            'UpperDeck_Walls': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.811321 0.811321 0.811321 1.000000'},
            'chair_table_aluminum_a_01': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.7 0.7 0.72 1.0'},
            'drain_mat_a_01': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.3 0.3 0.32 1.0'},
            'electric_generator_a_01': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.45 0.45 0.5 1.0'},
            'fire_extinguisher_a_01': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.8 0.2 0.2 1.0'},
            'pipe_insulated_a_01': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.6 0.6 0.7 1.0'},
            'pipe_insulated_b_01': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.5 0.5 0.6 1.0'},
            'shelving_metal_a_01': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.75 0.75 0.78 1.0'},
            'step_ladder_a_01': {'specular': '0.5', 'shininess': '0.2', 'rgba': '0.65 0.65 0.68 1.0'}
        }

        # Textures configuration from Ship2.xml
        textures = {
            'Base_metal': 'Base_metal.png',
            'compressor_basecolor': 'compressor_basecolor.png',
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
            texture.set('file', os.path.join(workspace_path, 'custom_env/ShipScene/ShipScene2/Ship', file))
            asset.append(texture)

        # Add materials
        for name, props in materials.items():
            material = Element('material')
            material.set('name', name)
            for key, value in props.items():
                material.set(key, value)
            asset.append(material)

        # Add meshes (28 meshes from Ship2.xml)
        for i in range(28):
            mesh_name = f'Ship_{i}'
            mesh = Element('mesh')
            mesh.set('file', os.path.join(workspace_path, f'custom_env/ShipScene/ShipScene2/Ship/{mesh_name}.obj'))
            mesh.set('name', mesh_name)
            mesh.set('scale', f'{self.scene_scale} {self.scene_scale} {self.scene_scale}')
            asset.append(mesh)

        self.world.root.append(asset)

    def _setup_scene(self):
        # Add gravity to the world
        option = Element('option')
        option.set('gravity', '0 0 -9.81')  # Standard gravity in negative Z direction
        self.world.root.append(option)
        
        ship_body = Element('body')
        ship_body.set('name', 'Ship')
        ship_body.set('pos', '5 5 -2.374')  # Match original position
        ship_body.set('euler', '1.57 3.14 0')  # Match original orientation
        
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
        
        # Scene meshes configuration from Ship2.xml
        visual_meshes = [
            ('Ship_0', 'LowerDeck_Walls'),
            ('Ship_1', 'step_ladder_a_01'),
            ('Ship_2', 'Engine'),
            ('Ship_3', 'Compressor'),
            ('Ship_4', 'Base_metal'),
            ('Ship_5', 'drain_mat_a_01'),
            ('Ship_6', 'UpperDeck_Walls'),
            ('Ship_7', 'DeckDetailed'),
            ('Ship_8', 'ShipControl'),
            ('Ship_9', 'Oxygen'),
            ('Ship_10', 'fire_extinguisher_a_01'),
            ('Ship_11', 'pipe_insulated_b_01'),
            ('Ship_12', 'pipe_insulated_a_01'),
            ('Ship_13', 'electric_generator_a_01'),
            ('Ship_14', 'Thermostat_Body_3'),
            ('Ship_15', 'Brass_-_Matte'),
            ('Ship_16', 'Stainless_Steel_-_Polished'),
            ('Ship_17', 'ThermostatBody_4'),
            ('Ship_18', 'Brass-Matte1'),
            ('Ship_19', 'Thermostat_Body_2'),
            ('Ship_20', 'ThermostatBody1'),
            ('Ship_21', 'shelving_metal_a_01'),
            ('Ship_22', 'chair_table_aluminum_a_01'),
            ('Ship_23', 'ControlboardDetailed'),
            ('Ship_24', 'M_Room_01'),
            ('Ship_25', 'ShipControl'),
            ('Ship_26', 'Device4'),
            ('Ship_27', 'Device1')
        ]

        # Add visual meshes
        for mesh_name, material in visual_meshes:
            geom = Element('geom')
            geom.set('mesh', mesh_name)
            geom.set('material', material)
            geom.set('group', '2')
            geom.set('contype', '0')
            geom.set('conaffinity', '0')
            geom.set('class', 'visual')
            ship_body.append(geom)

        # Add collision meshes ONLY for floor objects with proper properties
        floor_materials = ['DeckDetailed', 'drain_mat_a_01']
        for mesh_name, material in visual_meshes:
            if material in floor_materials:
                geom = Element('geom')
                geom.set('mesh', mesh_name)
                geom.set('group', '3')
                for key, value in floor_properties.items():
                    geom.set(key, value)
                ship_body.append(geom)

        self.world.worldbody.append(ship_body)

    def _setup_lighting(self):
        # Main directional light from above (from Ship2.xml)
        main_light = Element('light')
        main_light.set('name', 'top')
        main_light.set('pos', '0 0 10')
        main_light.set('dir', '0 0 -1')
        main_light.set('directional', 'true')
        main_light.set('diffuse', '0.7 0.7 0.7')
        main_light.set('specular', '0.3 0.3 0.3')
        main_light.set('ambient', '0.4 0.4 0.4')
        self.world.worldbody.append(main_light)

        # Fill light from the front (from Ship2.xml)
        fill_light = Element('light')
        fill_light.set('name', 'front')
        fill_light.set('pos', '5 0 3')
        fill_light.set('dir', '-1 0 -0.3')
        fill_light.set('directional', 'true')
        fill_light.set('diffuse', '0.4 0.4 0.4')
        fill_light.set('specular', '0.1 0.1 0.1')
        self.world.worldbody.append(fill_light)

        # Side light for depth (from Ship2.xml)
        side_light = Element('light')
        side_light.set('name', 'side')
        side_light.set('pos', '0 5 2')
        side_light.set('dir', '0 -1 -0.2')
        side_light.set('directional', 'true')
        side_light.set('diffuse', '0.3 0.3 0.3')
        side_light.set('specular', '0.1 0.1 0.1')
        self.world.worldbody.append(side_light)

    def _reset_robot(self, initial_pos=None):
        if initial_pos is None:
            initial_pos = {
                "pos": [-1.5, 0, 0.75],
                "yaw": -1.5708
            }
        elif isinstance(initial_pos, list):
            initial_pos = {
                "pos": initial_pos[:3],
                "yaw": initial_pos[5]
            }
        
        super()._reset_robot(initial_pos=initial_pos)

    def _check_success(self):
        self._success = False

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
