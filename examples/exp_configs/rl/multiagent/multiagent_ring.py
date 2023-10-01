"""Ring road example.

Trains a number of autonomous vehicles to stabilize the flow of 22 vehicles in
a variable length ring road.
"""
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.tune.registry import register_env

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.envs.multiagent import MultiAgentWaveAttenuationPOEnv
from flow.envs.multiagent import MultiAgentWaveAttenuationPONcomEnv
from flow.networks import RingNetwork
from flow.utils.registry import make_create_env

EXP_TAG = "model_valid-ring_road"
VERSION = 1
RENDER = False
ENV_NAME = MultiAgentWaveAttenuationPONcomEnv

# time horizon of a single rollout
HORIZON = 1000
# number of rollouts per training iteration
N_ROLLOUTS = 1
# number of parallel workers
N_CPUS = 1

vehicles = VehicleParams()
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=20,
)

flow_params = dict(
    exp_tag=EXP_TAG,
    env_name=ENV_NAME,
    network=RingNetwork,
    simulator='traci',
    sim=SumoParams(
        sim_step=0.1,
        render=RENDER,
    ),
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=0,
        additional_params={
            "max_accel": 3,
            "max_decel": 3,
            "ring_length": 230,
            "comm_distance": 20.0,
        },
    ),
    net=NetParams(
        additional_params={
            "length": 230,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 20,
        }, ),
    veh=vehicles,
    initial=InitialConfig(),
)

create_env, env_name = make_create_env(params=flow_params)
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
