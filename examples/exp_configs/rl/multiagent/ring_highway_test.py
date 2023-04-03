from flow.envs.multiagent import MultiAgentRingHighwayPOEnv
from flow.networks import RingHighwayNetwork
from flow.core.params import SumoParams, EnvParams, NetParams, VehicleParams, InitialConfig
from flow.core.params import SumoLaneChangeParams
from flow.controllers import ContinuousRouter, IDMController, RLController
from flow.utils.registry import make_create_env

# time horizon of a single rollout
HORIZON = 3000
N_ROLLOUTS = 1
N_CPUS = 10
# number of cooperative CAVs (<150)
N_CAV = 75
n_human = 150 - N_CAV

vehicles = VehicleParams()
vehicles.add(
  veh_id="rl",
  acceleration_controller=(RLController, {}),
  routing_controller=(ContinuousRouter, {}),
  num_vehicles=N_CAV
)
vehicles.add(
  veh_id="human",
  acceleration_controller=(IDMController, {"noise": 0.2}),
  routing_controller=(ContinuousRouter, {}),
  # "only_trategic_safe" or "sumo_default"
  lane_change_params=SumoLaneChangeParams(lane_change_mode="sumo_default"),
  num_vehicles=n_human
)

flow_params = dict(
  exp_tag='multiagent_ring_highway',
  env_name=MultiAgentRingHighwayPOEnv,
  network=RingHighwayNetwork,
  simulator='traci',
  sim=SumoParams(
    sim_step=0.1,
    render=True,
  ),
  env=EnvParams(
    horizon=HORIZON,
    warmup_steps=750,
    additional_params={
      'max_accel': 3,
      'max_decel': 3,
      'target_velocity': 100,
      'comm_distance': 20.0
    },
  ),
  net=NetParams(
    additional_params={
      "length": 2000,
      "speed_limit": 100,
      "resolution": 40,
      "connection_radius": 320,
      "merge_length": 200
    },
  ),
  veh=vehicles,
  initial=InitialConfig(bunching=20.0, spacing='custom'),
)

create_env, env_name = make_create_env(params=flow_params, version=0)

# use ray for testing
from ray.tune.registry import register_env
register_env(env_name, create_env)
test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
def gen_policy():
  return PPOTFPolicy, obs_space, act_space, {}

POLICY_GRAPHS = {'av': gen_policy()}

def policy_mapping_fn(_):
  return 'av'

POLICIES_TO_TRAIN = ['av']
