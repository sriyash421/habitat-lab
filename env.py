
import gym

import habitat_sim
import magnum as mn
import warnings
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
warnings.filterwarnings('ignore')
from habitat_sim.utils.settings import make_cfg
from matplotlib import pyplot as plt
from habitat_sim.utils import viz_utils as vut
from omegaconf import DictConfig
import numpy as np
from habitat.articulated_agents.robots import FetchRobot
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import ThirdRGBSensorConfig, HeadRGBSensorConfig, HeadPanopticSensorConfig
from habitat.config.default_structured_configs import SimulatorConfig, HabitatSimV0Config, AgentConfig
from habitat.config.default import get_agent_config
import habitat
from habitat_sim.physics import JointMotorSettings, MotionType
from omegaconf import OmegaConf

import git, os
repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
os.chdir(dir_path)



from habitat.config.default_structured_configs import TaskConfig, EnvironmentConfig, DatasetConfig, HabitatConfig
from habitat.config.default_structured_configs import ArmActionConfig, BaseVelocityActionConfig, OracleNavActionConfig, ActionConfig
from habitat.core.env import Env

def init_rearrange_sim(agent_dict):
    # Start the scene config
    sim_cfg = make_sim_cfg(agent_dict)    
    cfg = OmegaConf.create(sim_cfg)
    
    # Create the scene
    sim = RearrangeSim(cfg)

    # This is needed to initialize the agents
    sim.agents_mgr.on_new_scene()

    # For this tutorial, we will also add an extra camera that will be used for third person recording.
    camera_sensor_spec = habitat_sim.CameraSensorSpec()
    camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    camera_sensor_spec.uuid = "scene_camera_rgb"

    # TODO: this is a bit dirty but I think its nice as it shows how to modify a camera sensor...
    sim.add_sensor(camera_sensor_spec, 0)

    return sim

def make_sim_cfg(agent_dict):
    # Start the scene config
    sim_cfg = SimulatorConfig(type="RearrangeSim-v0")
    
    # Enable Horizon Based Ambient Occlusion (HBAO) to approximate shadows.
    sim_cfg.habitat_sim_v0.enable_hbao = True
    
    sim_cfg.habitat_sim_v0.enable_physics = True

    
    # Set up an example scene
    sim_cfg.scene = os.path.join(data_path, "hab3_bench_assets/hab3-hssd/scenes/103997919_171031233.scene_instance.json")
    sim_cfg.scene_dataset = os.path.join(data_path, "hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json")
    sim_cfg.additional_object_paths = [os.path.join(data_path, 'objects/ycb/configs/')]

    
    cfg = OmegaConf.create(sim_cfg)

    # Set the scene agents
    cfg.agents = agent_dict
    cfg.agents_order = list(cfg.agents.keys())
    return cfg

def make_hab_cfg(agent_dict, action_dict):
    sim_cfg = make_sim_cfg(agent_dict)
    task_cfg = TaskConfig(type="RearrangeEmptyTask-v0")
    task_cfg.actions = action_dict
    env_cfg = EnvironmentConfig()
    dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", data_path="data/hab3_bench_assets/episode_datasets/small_large.json.gz")
    
    
    hab_cfg = HabitatConfig()
    hab_cfg.environment = env_cfg
    hab_cfg.task = task_cfg
    hab_cfg.dataset = dataset_cfg
    hab_cfg.simulator = sim_cfg
    hab_cfg.simulator.seed = hab_cfg.seed

    return hab_cfg

def init_rearrange_env(agent_dict, action_dict):
    hab_cfg = make_hab_cfg(agent_dict, action_dict)
    res_cfg = OmegaConf.create(hab_cfg)
    return Env(res_cfg)


from habitat.tasks.rearrange.actions.articulated_agent_action import ArticulatedAgentAction
from habitat.core.registry import registry
from gym import spaces
import gzip
import json
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from habitat.datasets.rearrange.samplers.object_sampler import ObjectSampler

@registry.register_task_action
class PickObjIdAction(ArticulatedAgentAction):
    
    @property
    def action_space(self):
        MAX_OBJ_ID = 1000
        return spaces.Dict({
            f"{self._action_arg_prefix}pick_obj_id": spaces.Discrete(MAX_OBJ_ID)
        })

    def step(self, *args, **kwargs):
        obj_id = kwargs[f"{self._action_arg_prefix}pick_obj_id"]
        print(self.cur_grasp_mgr, obj_id)
        self.cur_grasp_mgr.snap_to_obj(obj_id)


class RearrangeEnv():
    def __init__(self) -> None:
        super().__init__()
        # Define the agent configuration
        main_agent_config = AgentConfig(is_set_start_state=True)
        urdf_path = os.path.join(data_path, "robots/hab_fetch/robots/hab_fetch.urdf")
        main_agent_config.articulated_agent_urdf = urdf_path
        main_agent_config.articulated_agent_type = "FetchRobot"

        # Define sensors that will be attached to this agent, here a third_rgb sensor and a head_rgb.
        # We will later talk about why we are giving the sensors these names
        main_agent_config.sim_sensors = {
            "third_rgb": ThirdRGBSensorConfig(),
            "head_rgb": HeadRGBSensorConfig(),
        }

        # We create a dictionary with names of agents and their corresponding agent configuration
        agent_dict = {"main_agent": main_agent_config}

        action_dict = {
            "pick_obj_id_action": ActionConfig(type="PickObjIdAction"),
            "base_velocity_action": BaseVelocityActionConfig(),
            "oracle_coord_action": OracleNavActionConfig(type="OracleNavCoordinateAction", spawn_max_dist_to_obj=1.0)
        }
        self.env = init_rearrange_env(agent_dict, action_dict)
        
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        camera_sensor_spec.uuid = "scene_camera_rgb"

        # TODO: this is a bit dirty but I think its nice as it shows how to modify a camera sensor...
        self.env.sim.add_sensor(camera_sensor_spec, 0)
        
        color_sensor_3rd_person_spec = habitat_sim.CameraSensorSpec()
        color_sensor_3rd_person_spec.uuid = "color_sensor_3rd_person"
        color_sensor_3rd_person_spec.sensor_type = habitat_sim.SensorType.COLOR
        
        color_sensor_3rd_person_spec.resolution = [
            1024,
            1024,
        ]
        color_sensor_3rd_person_spec.position = [
            1.0,
            8 + 0.2,
            -5.0,
        ]
        import math
        color_sensor_3rd_person_spec.orientation = [-math.pi / 2, 0.0, 0.0]
        color_sensor_3rd_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        # sensor_specs.append(color_sensor_3rd_person_spec)
        self.env.sim.add_sensor(color_sensor_3rd_person_spec, 0)

        # Here you can specify the amount of displacement in a forward action and the turn angle
        # agent_cfg = habitat_sim.agent.AgentConfiguration()
        # agent_cfg.sensor_specifications = sensor_specs
        # main_agent_config = AgentConfig()
        # urdf_path = os.path.join(data_path, "robots/hab_fetch/robots/hab_fetch.urdf")
        # main_agent_config.articulated_agent_urdf = urdf_path
        # main_agent_config.articulated_agent_type = "FetchRobot"

        # # Define sensors that will be attached to this agent, here a third_rgb sensor and a head_rgb.
        # # We will later talk about why we are giving the sensors these names
        # main_agent_config.sim_sensors = {
        #     "third_rgb": ThirdRGBSensorConfig(),
        #     "head_rgb": HeadRGBSensorConfig(),
        # }

        # # We create a dictionary with names of agents and their corresponding agent configuration
        # agent_dict = {"main_agent": main_agent_config}
        # self.sim = init_rearrange_sim(agent_dict)
        
        # self.num_goals = 5
        
        # fps = 60 # Default value for make video
        # self.dt = 1./fps
        
        # self.goal_location = mn.Vector(-7.16318, 0.85088, -3.13086)
        episode_file = os.path.join(data_path, "hab3_bench_assets/episode_datasets/small_large.json.gz")
        # Load the dataset
        with gzip.open(episode_file, "rt") as f: 
            episode_files = json.loads(f.read())

        # Get the first episode
        episode = episode_files["episodes"][2]
        self.rearrange_episode = RearrangeEpisode(**episode)
    
    def reset(self):
        self.env.reset()
        
        sim = self.env.sim
        art_agent = sim.articulated_agent
        art_agent._fixed_base = True
        sim.agents_mgr.on_new_scene()

        sim.reconfigure(sim.habitat_config, ep_info=self.rearrange_episode)
        
        # camera_sensor_spec = habitat_sim.CameraSensorSpec()
        # camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        # camera_sensor_spec.uuid = "scene_camera_rgb"

        # # TODO: this is a bit dirty but I think its nice as it shows how to modify a camera sensor...
        # sim.add_sensor(camera_sensor_spec, 0)
        
        art_agent = self.env.sim.articulated_agent
        init_pos = mn.Vector3(-5.5,0,-1.5)
        # Define the agent configuration
        # sim = self.env.sim
        sim.reset()
        art_agent.sim_obj.motion_type = MotionType.KINEMATIC
        sim.articulated_agent.base_pos =  init_pos 
        _ = sim.step({})

        # art_agent._fixed_base = False
        # self.sim.agents_mgr.on_new_scene()
        # # The base is not fixed anymore
        # art_agent.sim_obj.motion_type = MotionType.DYNAMIC
        # art_agent.base_pos = init_pos + mn.Vector3(0,1.5,0)
        # _ = self.sim.step({})
        # observations = self.sim.get_sensor_observations()
        # # return observations
        # self.frames = []
    
    def _step(self, action_dict):
        self.env.step(action_dict)
        obs = self.env.sim.get_sensor_observations()
        done = self.env.episode_over
        return obs, done
    
    def navigate_to_pos(self, object_trans):
        observations = []
        done = False
        object_agent_vec = self.env.sim.articulated_agent.base_pos - object_trans
        object_agent_vec.y = 0
        dist_agent_object = object_agent_vec.length()
        # Walk towards the object

        agent_displ = np.inf
        agent_rot = np.inf
        prev_rot = self.env.sim.articulated_agent.base_rot
        prev_pos = self.env.sim.articulated_agent.base_pos
        while agent_displ > 1e-9 or agent_rot > 1e-9:
            prev_rot = self.env.sim.articulated_agent.base_rot
            prev_pos = self.env.sim.articulated_agent.base_pos
            action_dict = {
                "action": ("oracle_coord_action"), 
                "action_args": {
                    "oracle_nav_lookat_action": object_trans,
                    "mode": 1
                }
            }
            obs, done = self._step(action_dict)
            observations.append(obs)
            if done:
                break
            cur_rot = self.env.sim.articulated_agent.base_rot
            cur_pos = self.env.sim.articulated_agent.base_pos
            agent_displ = (cur_pos - prev_pos).length()
            agent_rot = np.abs(cur_rot - prev_rot)

        # Wait
        # for _ in range(20):
        #     if done:
        #         break
        #     action_dict = {"action": (), "action_args": {}}
        #     obs, done = self.env.step()
        #     observations.append(self.env.step(action_dict))
        return observations, done
            
    def pick_object(self, object_id):
        rom = self.env.sim.get_rigid_object_manager()
        first_object = rom.get_object_by_id(object_id)

        object_trans = first_object.translation
        print(first_object.handle, "is in", object_trans)

        # print(sample)
        observations, done = self.navigate_to_pos(object_trans)
        # delta = 2.0
        if done:
            return observations, "cannot navigate to object"

        action_dict = {"action": ("pick_obj_id_action"), "action_args": {"pick_obj_id": object_id}}
        obs, done = self._step(action_dict)
        observations.append(obs)
        
        if done:
            return observations, "failed to pick object"
        # for _ in range(10):
        #     action_dict = {"action": (), "action_args": {}}
        #     observations.append(self.env.step(action_dict))
            
        return observations, "picked object"

    def place_object(self, goal):
        # rom = self.env.sim.get_rigid_object_manager()
        # first_object = rom.get_object_by_id(goal_id)

        # object_trans = first_object.translation
        observations, done = self.navigate_to_pos(goal)
        print(f"Goal:", goal)
        if done:
            return observations, "cannot navigate to goal"
        
        agent_id = 0
        grasp_manager = self.env.sim.agents_mgr[agent_id].grasp_mgrs[0]
        grasp_manager.desnap()
        for _ in range(10):
            action_dict = {"action": (), "action_args": {}}
            obs, done = self._step(action_dict)
            observations.append(obs)
            if done:
                return observations, "cannot place object"
        return observations, "placed object successfull"
        
    def step(self, obj_index):
        object_maps = [
            # "003_cracker_box_:0000",
            # "008_pudding_box_:0000",
            # "009_gelatin_box_:0000",
            # "008_pudding_box_:0001",
            # "007_tuna_fish_can_:0000",
            # "007_tuna_fish_can_:0001",
            # "003_cracker_box_:0001",
            # "007_tuna_fish_can_:0002",
            # "005_tomato_soup_can_:0000"
            # "004_sugar_box_:0000",
            # "009_gelatin_box_:0000",
            # "010_potted_meat_can_:0000",
            # # "024_bowl_:0000",
            # "007_tuna_fish_can_:0000",
            # "002_master_chef_can_:0000",
            # "024_bowl_:0000",
            "010_potted_meat_can_:0000",
            "009_gelatin_box_:0001",
            "009_gelatin_box_:0000",
            "008_pudding_box_:0000",
            "008_pudding_box_:0001",
            "002_master_chef_can_:0000",
            # "002_master_chef_can_:0001",
            "008_pudding_box_:0002",
            "005_tomato_soup_can_:0000",
        ]
        
        # object_maps = [
        #     (111, "003_cracker_box_"),
        #     (113, "004_sugar_box_"),
        #     (117, "009_gelatin_box_"),
        #     (118, "010_potted_meat_can_"),
        #     (114, "007_tuna_fish_can_"),
        #     (109, "002_master_chef_can_"),
        # ]
        rom = self.env.sim.get_rigid_object_manager()
        obj_id = rom.get_object_id_by_handle(object_maps[obj_index])
        
        first_object = rom.get_object_by_id(obj_id)
        print(f"chosen: {object_maps[obj_index]}, selected: {first_object.handle}")
        goal = mn.Vector3(-3.29429, 0.894202, -7.25061)#mn.Vector3(-7.16318, 0.85088, -3.13086)
        # {}, {-6.25845, 0.898544, -5.20799}
        # Vector(-7.07397, 0.180179, -6.73884)
        # goal_obj_id = rom.get_object_id_by_handle("024_bowl_:0000")
        observations, state1 = self.pick_object(obj_id)
        print(state1)
        observations2, state2 = self.place_object(goal)
        observations.extend(observations2)
        print(state2)
        return observations, object_maps[obj_index]
        

def print_env(env):
    sim = env.env.sim
    # aom = sim.get_articulated_object_manager()
    rom = sim.get_rigid_object_manager()

    # We can query the articulated and rigid objects

    # print("List of articulated objects:")
    # for handle, ao in aom.get_objects_by_handle_substring().items():
    #     print(handle, "id", aom.get_object_id_by_handle(handle))

    print("\nList of rigid objects:")
    obj_ids = []
    for handle, ro in rom.get_objects_by_handle_substring().items():
        if ro.awake:
            print(handle, "id", ro.object_id)
            obj_ids.append(ro.object_id)
    print("=="*10)


env = RearrangeEnv()

for obj_id in range(9):
    env.reset()
    print_env(env)
    observations, obj_name = env.step(obj_id)
    vut.make_video(
        observations,
        "color_sensor_3rd_person",
        "color",
        f"videos2/{obj_name}",
        open_vid=False,
    )






