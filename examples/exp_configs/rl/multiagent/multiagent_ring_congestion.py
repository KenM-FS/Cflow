"""Ring road example.

Trains a number of autonomous vehicles to stabilize the flow of 22 vehicles in
a variable length ring road.
"""
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.tune.registry import register_env

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoLaneChangeParams
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.envs.multiagent import MultiAgentRingHighwayPOCommEnv
from flow.envs.multiagent import MultiAgentRingHighwayPONcomEnv
from flow.networks import RingNetwork
from flow.utils.registry import make_create_env

EXP_TAG = "learningCAV"
VERSION = 5
RENDER = False
ENV_NAME = MultiAgentRingHighwayPOCommEnv
# ENV_NAME = MultiAgentRingHighwayPONcomEnv

RATE_CAV = 0.3
NUM_LANE = 3

# time horizon of a single rollout
HORIZON = 1000
# number of rollouts per training iteration
N_ROLLOUTS = 1
# number of parallel workers
N_CPUS = 2
# number of automated vehicles. Must be less than or equal to 22.
NUM_TOTAL = 150

num_cav = round(NUM_TOTAL * RATE_CAV)
exp_tag = EXP_TAG + "_" + str(RATE_CAV) + "_" + str(num_cav)

# We evenly distribute the automated vehicles in the network.
set_lane = round(NUM_TOTAL / NUM_LANE)
set_cav = round(num_cav / NUM_LANE)
set_human = set_lane - set_cav
set_human_remaining = set_human

vehicles = VehicleParams()
for i in range(set_cav):
    # Add one automated vehicle.
    vehicles.add(
        veh_id="rl_{}".format(i),
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=3
    )

    # Add a fraction of the remaining human vehicles.
    sets_to_add = round(set_human_remaining / (set_cav - i))
    set_human_remaining -= sets_to_add

    vehicles.add(
        veh_id="human_{}".format(i),
        acceleration_controller=(IDMController, {
            "noise": 7.0
        }),
        lane_change_params=SumoLaneChangeParams(lane_change_mode="sumo_default"),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=sets_to_add*3)


flow_params = dict(
    exp_tag=exp_tag,
    env_name=ENV_NAME,
    network=RingNetwork,
    simulator='traci',
    sim=SumoParams(
        sim_step=0.1,
        render=RENDER,
        overtake_right=True,
    ),
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=200,
        additional_params={
            "max_accel": 3,
            "max_decel": 3,
            "target_velocity": 100,
            "comm_distance": 20.0,
        },
    ),
    net=NetParams(
        additional_params={
            "length": 2000,
            "lanes": NUM_LANE,
            "speed_limit": 100,
            "resolution": 40,
        }, ),
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)


create_env, env_name = make_create_env(params=flow_params, version=VERSION-1)

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


def gen_policy():
    """Generate a policy in RLlib."""
    return PPOTFPolicy, obs_space, act_space, {}


# Setup PG with an ensemble of `num_policies` different policy graphs
POLICY_GRAPHS = {'av': gen_policy()}


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'av'


POLICIES_TO_TRAIN = ['av']
