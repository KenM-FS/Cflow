import numpy as np
from gym.spaces import Box
from gym.spaces.tuple import Tuple
from flow.envs.multiagent.base import MultiEnv
import copy

ADDITIONAL_ENV_PARAMS = {
  # in m/s^2
  'max_accel': 3,
  'max_decel': 3,
  'target_velocity': 100,
}

class MultiAgentRingFOCommEnv(MultiEnv):

  def __init__(self, env_params, sim_params, network, simulator='traci'):
    for p in ADDITIONAL_ENV_PARAMS.keys():
      if p not in env_params.additional_params:
        raise KeyError(
          'Environment parameter "{}" not supplied'.format(p)
        )

    super().__init__(env_params, sim_params, network, simulator)

  @property
  def observation_space(self):
    # Obs space for information from communication
    basic_obs = Tuple((
      Box(low=-1.,          high=1.,          shape=(1,), dtype=np.float16),
      Box(low=-float(2000), high=float(2000), shape=(3,), dtype=np.float16),
      Box(low=float(0),     high=float(200),  shape=(3,), dtype=np.float16),
      Box(low=-float(2000), high=float(2000), shape=(3,), dtype=np.float16),
      Box(low=float(0),     high=float(200),  shape=(3,), dtype=np.float16)
    ))

    # Observation space for a vehicle
    obs_space = Tuple((
      Box(low=-1.,          high=1.,          shape=(1,), dtype=np.float16), # Velocity/SpeedLimit
      Box(low=-float(2000), high=float(2000), shape=(3,), dtype=np.float16), # Headway
      Box(low=float(0),     high=float(200),  shape=(3,), dtype=np.float16), # Leader velocity
      Box(low=-float(2000), high=float(2000), shape=(3,), dtype=np.float16), # Tailway
      Box(low=float(0),     high=float(200),  shape=(3,), dtype=np.float16), # Follower velocity
      Tuple((basic_obs,) * self.initial_vehicles.num_rl_vehicles)
    ))
    return obs_space

  @property
  def action_space(self):
    return Tuple((
      Box(
        low=-np.abs(self.env_params.additional_params['max_decel']),
        high=self.env_params.additional_params['max_accel'],
        shape=(1,),
        dtype=np.float16
      ),
      # Discrete(3)
      Box(
        low=0,
        high=2,
        shape=(1,),
        dtype=np.int8
      )
    ))

  def _apply_rl_actions(self, rl_actions):
    if rl_actions:
      for rl_id, actions in rl_actions.items():
        accel = actions[0][0]
        lane_change = int(actions[1] - 1)

        self.k.vehicle.apply_acceleration(rl_id, accel)
        self.k.vehicle.apply_lane_change(rl_id, lane_change)

  def get_state(self):
    obs = {}
    rl_ids = self.k.vehicle.get_rl_ids()

    for rl_id in rl_ids:
      edge = self.k.vehicle.get_edge(rl_id)
      speed_limit = self.k.network.speed_limit(edge)
      speed = self.k.vehicle.get_speed(rl_id)

      headway = np.array(self.k.vehicle.get_lane_headways(rl_id), dtype=np.float16)
      leader_velocity = np.array(self.k.vehicle.get_lane_leaders_speed(rl_id), dtype=np.float16)
      tailway = np.array(self.k.vehicle.get_lane_tailways(rl_id), dtype=np.float16)
      follower_velocity = np.array(self.k.vehicle.get_lane_followers_speed(rl_id), dtype=np.float16)

      observation = [
        np.array([speed / speed_limit], dtype=np.float16),
        headway.copy(),
        leader_velocity.copy(),
        tailway.copy(),
        follower_velocity.copy()
      ]
      obs.update({rl_id: observation})

    base_info = (
      np.array([0.], dtype=np.float16),
      np.array([0., 0., 0.], dtype=np.float16),
      np.array([0., 0., 0.], dtype=np.float16),
      np.array([0., 0., 0.], dtype=np.float16),
      np.array([0., 0., 0.], dtype=np.float16)
    )
    states = {}
    for rl_id in rl_ids:
      # state = copy.deepcopy(obs[rl_id])
      state = copy.copy(obs[rl_id])

      # # Reset own information through comm
      # obs_copy = obs.copy()
      # obs_copy[rl_id] = list(base_info)

      # comm_info = [tuple(ob) for ob in obs_copy.values()]
      comm_info = [tuple(ob) for ob in obs.values()]
      state.append(tuple(comm_info.copy()))
      states[rl_id] = tuple(state.copy())
    return states

  def compute_reward(self, rl_actions, **kwargs):
    # in the warmup steps
    if rl_actions is None:
      return{}

    vel = np.array([
      self.k.vehicle.get_speed(veh_id)
      for veh_id in self.k.vehicle.get_ids()
    ])

    # safety first
    if any(vel < -100) or kwargs['fail']:
      return 0

    rewards = {}
    # reward by average velocity of all vehicles
    eta_1 = 1
    reward_global = np.mean(vel) * eta_1

    # reward by target velocity of each agent
    eta_3 = 1
    for rl_id in self.k.vehicle.get_rl_ids():
      edge = self.k.vehicle.get_edge(rl_id)
      speed_limit = self.k.network.speed_limit(edge)
      speed = self.k.vehicle.get_speed(rl_id)
      rewards[rl_id] = max(reward_global + (speed/speed_limit)*100*eta_3, 0)

    return rewards
