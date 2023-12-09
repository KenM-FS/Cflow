import numpy as np
import math
from gym.spaces import Box
from gym.spaces.dict import Dict
from gym.spaces.discrete import Discrete
from gym.spaces.tuple import Tuple
from flow.envs.multiagent.base import MultiEnv
import copy

ADDITIONAL_ENV_PARAMS = {
  # in m/s^2
  'max_accel': 3,
  'max_decel': 3,
  'target_velocity': 100,
  # 'target_velocity_main': 100,
  # 'target_velocity_merge': 60,
  'comm_distance': 20.0
}

# MultiAgentRingHighwayPOCommEnv
class MultiAgentRingHighwayPONcomEnv(MultiEnv):

  def __init__(self, env_params, sim_params, network, simulator='traci'):
    for p in ADDITIONAL_ENV_PARAMS.keys():
      if p not in env_params.additional_params:
        raise KeyError(
          'Environment parameter "{}" not supplied'.format(p)
        )

    super().__init__(env_params, sim_params, network, simulator)

  @property
  def observation_space(self):
    obs_space = Tuple((
      Box(low=-1.,           high=1.,           shape=(1,)),
      Box(low=-float('inf'), high=float('inf'), shape=(4,)),
      Box(low=-float('inf'), high=float('inf'), shape=(4,)),
      Box(low=-float('inf'), high=float('inf'), shape=(4,)),
      Box(low=-float('inf'), high=float('inf'), shape=(4,))
    ))
    return obs_space

  @property
  def action_space(self):
    return Tuple((
      Box(
        low=-np.abs(self.env_params.additional_params['max_decel']),
        high=self.env_params.additional_params['max_accel'],
        shape=(1,),
        dtype=np.float32
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

      headway = np.array(self.k.vehicle.get_lane_headways(rl_id), dtype=np.float32)
      if headway.shape == (3,):
        headway = np.append(headway, np.array([0.], dtype=np.float32))
      elif headway.shape == (1,):
        headway = np.append(headway, np.array([0., 0., 0.], dtype=np.float32))

      leader_velocity = np.array(self.k.vehicle.get_lane_leaders_speed(rl_id), dtype=np.float32)
      if leader_velocity.shape == (3,):
        leader_velocity = np.append(leader_velocity, np.array([0.], dtype=np.float32))
      elif leader_velocity.shape == (1,):
        leader_velocity = np.append(leader_velocity, np.array([0., 0., 0.], dtype=np.float32))

      tailway = np.array(self.k.vehicle.get_lane_tailways(rl_id), dtype=np.float32)
      if tailway.shape == (3,):
        tailway = np.append(tailway, np.array([0.], dtype=np.float32))
      elif tailway.shape == (1,):
        tailway = np.append(tailway, np.array([0., 0., 0.], dtype=np.float32))

      follower_velocity = np.array(self.k.vehicle.get_lane_followers_speed(rl_id), dtype=np.float32)
      if follower_velocity.shape == (3,):
        follower_velocity = np.append(follower_velocity, np.array([0.], dtype=np.float32))
      elif follower_velocity.shape == (1,):
        follower_velocity = np.append(follower_velocity, np.array([0., 0., 0.], dtype=np.float32))

      observation = [
        np.array([speed / speed_limit], dtype=np.float32),
        headway.copy(),
        leader_velocity.copy(),
        tailway.copy(),
        follower_velocity.copy()
      ]
      obs.update({rl_id: observation})

    return obs

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
    ## punish accelerations (should lead to reduce stop-and-go wave)
    # eta_2 = 2
    # accel_action = 0
    # if len(rl_actions) > 0:
    #   accel_action = np.array(list(rl_actions.values()))[:, 0]
    # mean_actions = np.mean(np.abs(accel_action))
    # accel_threshold = 0
    # if mean_actions > accel_threshold:
    #   reward_global += eta_2 * (accel_threshold - mean_actions)

    # reward by target velocity of each agent
    eta_3 = 1
    for rl_id in self.k.vehicle.get_rl_ids():
      edge = self.k.vehicle.get_edge(rl_id)
      speed_limit = self.k.network.speed_limit(edge)
      speed = self.k.vehicle.get_speed(rl_id)
      rewards[rl_id] = max(reward_global + (speed/speed_limit)*100*eta_3, 0)

    return rewards
class MultiAgentRingHighwayPOCommEnv(MultiAgentRingHighwayPONcomEnv):

  def __init__(self, env_params, sim_params, network, simulator='traci'):
    for p in ADDITIONAL_ENV_PARAMS.keys():
      if p not in env_params.additional_params:
        raise KeyError(
          'Environment parameter "{}" not supplied'.format(p)
        )

    super().__init__(env_params, sim_params, network, simulator)

  @property
  def observation_space(self):
    basic_obs = Tuple((
      Box(low=-1.,           high=1.,           shape=(1,)),
      Box(low=-float('inf'), high=float('inf'), shape=(4,)),
      Box(low=-float('inf'), high=float('inf'), shape=(4,)),
      Box(low=-float('inf'), high=float('inf'), shape=(4,)),
      Box(low=-float('inf'), high=float('inf'), shape=(4,))
    ))
    obs_space = Tuple((
      Box(low=-1.,           high=1.,           shape=(1,)),
      Box(low=-float('inf'), high=float('inf'), shape=(4,)),
      Box(low=-float('inf'), high=float('inf'), shape=(4,)),
      Box(low=-float('inf'), high=float('inf'), shape=(4,)),
      Box(low=-float('inf'), high=float('inf'), shape=(4,)),
      Tuple((basic_obs,) * self.initial_vehicles.num_rl_vehicles)
    ))
    return obs_space

  def get_state(self):
    obs = {}
    comm_distance = self.env_params.additional_params['comm_distance']
    rl_ids = self.k.vehicle.get_rl_ids()
    one_matrix = dict(zip(rl_ids, [0]*len(rl_ids)))
    comm_matrix = {}
    for rl_id in rl_ids: comm_matrix[rl_id] = one_matrix.copy()
    for i in rl_ids:
      for j in rl_ids:
        if i == j: continue
        i_coord = self.k.vehicle.get_2d_position(i)
        j_coord = self.k.vehicle.get_2d_position(j)
        if self.is_closer(i_coord, j_coord, comm_distance):
          comm_matrix[i][j] = 1
          comm_matrix[j][i] = 1

    for rl_id in rl_ids:
      queue = [id for id, v in comm_matrix[rl_id].items() if v == 1]
      while len(queue) > 0:
        for relayed_id, v in comm_matrix[queue.pop(0)].items():
          if relayed_id == rl_id: continue
          elif v == 1 and comm_matrix[rl_id][relayed_id] != 1:
            comm_matrix[rl_id][relayed_id] = 1
            queue.append(relayed_id)

    for rl_id in rl_ids:
      edge = self.k.vehicle.get_edge(rl_id)
      speed_limit = self.k.network.speed_limit(edge)
      speed = self.k.vehicle.get_speed(rl_id)

      headway = np.array(self.k.vehicle.get_lane_headways(rl_id), dtype=np.float32)
      if headway.shape == (3,):
        headway = np.append(headway, np.array([0.], dtype=np.float32))
      elif headway.shape == (1,):
        headway = np.append(headway, np.array([0., 0., 0.], dtype=np.float32))

      leader_velocity = np.array(self.k.vehicle.get_lane_leaders_speed(rl_id), dtype=np.float32)
      if leader_velocity.shape == (3,):
        leader_velocity = np.append(leader_velocity, np.array([0.], dtype=np.float32))
      elif leader_velocity.shape == (1,):
        leader_velocity = np.append(leader_velocity, np.array([0., 0., 0.], dtype=np.float32))

      tailway = np.array(self.k.vehicle.get_lane_tailways(rl_id), dtype=np.float32)
      if tailway.shape == (3,):
        tailway = np.append(tailway, np.array([0.], dtype=np.float32))
      elif tailway.shape == (1,):
        tailway = np.append(tailway, np.array([0., 0., 0.], dtype=np.float32))

      follower_velocity = np.array(self.k.vehicle.get_lane_followers_speed(rl_id), dtype=np.float32)
      if follower_velocity.shape == (3,):
        follower_velocity = np.append(follower_velocity, np.array([0.], dtype=np.float32))
      elif follower_velocity.shape == (1,):
        follower_velocity = np.append(follower_velocity, np.array([0., 0., 0.], dtype=np.float32))

      observation = [
        np.array([speed / speed_limit], dtype=np.float32),
        headway.copy(),
        leader_velocity.copy(),
        tailway.copy(),
        follower_velocity.copy()
      ]
      obs.update({rl_id: observation})

    vel_mean = np.mean(np.array([
      self.k.vehicle.get_speed(veh_id)
      for veh_id in self.k.vehicle.get_ids()
    ]))
    # base_info = (
    #   np.array([(
    #   vel_mean / self.env_params.additional_params['target_velocity']
    #   )], dtype=np.float32),
    #   np.array([9999, 9999, 9999, 9999], dtype=np.float32),
    #   np.array([vel_mean, vel_mean, vel_mean, vel_mean], dtype=np.float32),
    #   np.array([9999, 9999, 9999, 9999], dtype=np.float32),
    #   np.array([vel_mean, vel_mean, vel_mean, vel_mean], dtype=np.float32)
    # )
    base_info = (
      np.array([0.], dtype=np.float16),
      np.array([0., 0., 0.], dtype=np.float16),
      np.array([0., 0., 0.], dtype=np.float16),
      np.array([0., 0., 0.], dtype=np.float16),
      np.array([0., 0., 0.], dtype=np.float16)
    )
    states = {}
    for rl_id in rl_ids:
      state = copy.copy(obs[rl_id])
      # comm_info = {}
      # needs to be initialized because num of rl vehs will change in time
      # (vehs will be added in time)
      comm_info = []
      for i in range(self.initial_vehicles.num_rl_vehicles):
        comm_info.append(copy.copy(base_info))

      count = 0
      for comm_id, v in comm_matrix[rl_id].items():
        if v == 1:
          comm_info[count] = tuple(obs[comm_id].copy())
        count += 1
      state.append(tuple(comm_info.copy()))
      states[rl_id] = tuple(state.copy())
    return states

  def is_closer(self, i, j, threshold):
    distance = math.sqrt((i[0] - i[0])**2 + (i[1] - j[1])**2)
    return distance < threshold
