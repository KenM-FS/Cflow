from flow.envs.multiagent import MultiAgentRingHighwayPOCommEnv
from flow.networks import RingHighwayNetwork
from flow.networks.ring_highway import ADDITIONAL_NET_PARAMS
from flow.core.params import SumoParams, EnvParams, NetParams, VehicleParams, InitialConfig, SumoLaneChangeParams
from flow.controllers import ContinuousRouter, IDMController, RLController
from flow.utils.registry import make_create_env
from ray.tune.registry import register_env
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy

"""
python3 flow/examples/train.py multiagent_ring_highway --num_steps=500
"""
EXP_TAG = "less_lane_change"
VERSION = 1

HORIZON = 1000
N_CPUS = 1
N_ROLLOUTS = 1

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {"noise": 0.2}),
    routing_controller=(ContinuousRouter, {}),
    lane_change_params=SumoLaneChangeParams(lane_change_mode="sumo_default"),
    num_vehicles=30,
)

vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=30,
)

flow_params = dict(
    exp_tag=EXP_TAG,
    env_name=MultiAgentRingHighwayPOCommEnv,
    network=RingHighwayNetwork,
    simulator='traci',
    sim=SumoParams(
        sim_step=0.1,
        render=False,
    ),
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=0,
        additional_params={
            'max_accel': 3,
            'max_decel': 3,
            'target_velocity': 100,
            'comm_distance': 20.0,
        },
    ),
    net=NetParams(
        additional_params=ADDITIONAL_NET_PARAMS
    ),
    veh=vehicles,
    initial=InitialConfig(spacing="uniform", perturbation=1),
)

create_env, env_name = make_create_env(params=flow_params, version=VERSION)
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space

def gen_policy():
    return PPOTFPolicy, obs_space, act_space, {}

POLICY_GRAPHS = {'av': gen_policy()}

def policy_mapping_fn(_):
    return 'av'

POLICIES_TO_TRAIN = ['av']
