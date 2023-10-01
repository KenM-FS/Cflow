"""
Environment used to train a stop-and-go dissipating controller.

This is the environment that was used in:

C. Wu, A. Kreidieh, K. Parvate, E. Vinitsky, A. Bayen, "Flow: Architecture and
Benchmarking for Reinforcement Learning in Traffic Control," CoRR, vol.
abs/1710.05465, 2017. [Online]. Available: https://arxiv.org/abs/1710.05465
"""

import numpy as np
from gym.spaces.box import Box
from gym.spaces.tuple import Tuple
import random
from scipy.optimize import fsolve
from copy import deepcopy

from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.envs.multiagent.base import MultiEnv
from flow.envs.ring.wave_attenuation import v_eq_max_function


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 3,
    # maximum deceleration of autonomous vehicles
    'max_decel': 3,
    # bounds on the ranges of ring road lengths the autonomous vehicle is
    # trained on
    'ring_length': 230,
    'comm_distance': 20.0
}


class MultiWaveAttenuationPOEnv(MultiEnv):
    """Multiagent shared model version of WaveAttenuationPOEnv.

    Intended to work with Lord Of The Rings Network.
    Note that this environment current
    only works when there is one autonomous vehicle
    on each ring.

    Required from env_params: See parent class

    States
        See parent class

    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class
    """

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        add_params = self.net_params.additional_params
        num_rings = add_params['num_rings']
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(int(self.initial_vehicles.num_rl_vehicles / num_rings), ),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        obs = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

            # normalizers
            max_speed = 15.
            max_length = self.env_params.additional_params['ring_length'][1]

            observation = np.array([
                self.k.vehicle.get_speed(rl_id) / max_speed,
                (self.k.vehicle.get_speed(lead_id) -
                 self.k.vehicle.get_speed(rl_id))
                / max_speed,
                self.k.vehicle.get_headway(rl_id) / max_length
            ])
            obs.update({rl_id: observation})

        return obs

    def _apply_rl_actions(self, rl_actions):
        """Split the accelerations by ring."""
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            accel = list(rl_actions.values())
            self.k.vehicle.apply_acceleration(rl_ids, accel)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rew = {}
        for rl_id in rl_actions.keys():
            edge_id = rl_id.split('_')[1]
            edges = self.gen_edges(edge_id)
            vehs_on_edge = self.k.vehicle.get_ids_by_edge(edges)
            vel = np.array([
                self.k.vehicle.get_speed(veh_id)
                for veh_id in vehs_on_edge
            ])
            if any(vel < -100) or kwargs['fail']:
                return 0.

            target_vel = self.env_params.additional_params['target_velocity']
            max_cost = np.array([target_vel] * len(vehs_on_edge))
            max_cost = np.linalg.norm(max_cost)

            cost = vel - target_vel
            cost = np.linalg.norm(cost)

            rew[rl_id] = max(max_cost - cost, 0) / max_cost
        return rew

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id
            self.k.vehicle.set_observed(lead_id)

    @staticmethod
    def gen_edges(i):
        """Return the edges corresponding to the rl id."""
        return ['top_{}'.format(i), 'left_{}'.format(i),
                'right_{}'.format(i), 'bottom_{}'.format(i)]

class MultiAgentWaveAttenuationPOEnv(MultiEnv):
    """Multi-agent variant of WaveAttenuationPOEnv.

    Required from env_params:

    * max_accel: maximum acceleration of autonomous vehicles
    * max_decel: maximum deceleration of autonomous vehicles
    * ring_length: bounds on the ranges of ring road lengths the autonomous
      vehicle is trained on. If set to None, the environment sticks to the ring
      road specified in the original network definition.

    States
        The state of each agent (AV) consists of the speed and headway of the
        ego vehicle, as well as the difference in speed between the ego vehicle
        and its leader. There is no assumption on the number of vehicles in the
        network.

    Actions
        Actions are an acceleration for each rl vehicle, bounded by the maximum
        accelerations and decelerations specified in EnvParams.

    Rewards
        The reward function rewards high average speeds from all vehicles in
        the network, and penalizes accelerations by the rl vehicle. This reward
        is shared by all agents.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=-5, high=5, shape=(3,), dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1,),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        obs = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

            # normalizers
            max_speed = 15.
            max_length = self.env_params.additional_params['ring_length'][1]

            observation = np.array([
                self.k.vehicle.get_speed(rl_id) / max_speed,
                (self.k.vehicle.get_speed(lead_id) -
                 self.k.vehicle.get_speed(rl_id))
                / max_speed,
                self.k.vehicle.get_headway(rl_id) / max_length
            ])
            obs.update({rl_id: observation})

        return obs

    def _apply_rl_actions(self, rl_actions):
        """Split the accelerations by ring."""
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            accel = list(rl_actions.values())
            self.k.vehicle.apply_acceleration(rl_ids, accel)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.

        # reward average velocity
        eta_2 = 4.
        reward = eta_2 * np.mean(vel) / 20

        # punish accelerations (should lead to reduced stop-and-go waves)
        eta = 4  # 0.25
        mean_actions = np.mean(np.abs(list(rl_actions.values())))
        accel_threshold = 0

        if mean_actions > accel_threshold:
            reward += eta * (accel_threshold - mean_actions)

        return {key: reward for key in self.k.vehicle.get_rl_ids()}

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id
            self.k.vehicle.set_observed(lead_id)

    def reset(self, new_inflow_rate=None):
        """See parent class.

        The sumo instance is reset with a new ring length, and a number of
        steps are performed with the rl vehicle acting as a human vehicle.
        """
        # skip if ring length is None
        if self.env_params.additional_params['ring_length'] is None:
            return super().reset()

        # reset the step counter
        self.step_counter = 0

        # update the network
        initial_config = InitialConfig(bunching=50, min_gap=0)
        length = random.randint(
            self.env_params.additional_params['ring_length'][0],
            self.env_params.additional_params['ring_length'][1])
        additional_net_params = {
            'length':
                length,
            'lanes':
                self.net_params.additional_params['lanes'],
            'speed_limit':
                self.net_params.additional_params['speed_limit'],
            'resolution':
                self.net_params.additional_params['resolution']
        }
        net_params = NetParams(additional_params=additional_net_params)

        self.network = self.network.__class__(
            self.network.orig_name, self.network.vehicles,
            net_params, initial_config)
        self.k.vehicle = deepcopy(self.initial_vehicles)
        self.k.vehicle.kernel_api = self.k.kernel_api
        self.k.vehicle.master_kernel = self.k

        # solve for the velocity upper bound of the ring
        v_guess = 4
        v_eq_max = fsolve(v_eq_max_function, np.array(v_guess),
                          args=(len(self.initial_ids), length))[0]

        print('\n-----------------------')
        print('ring length:', net_params.additional_params['length'])
        print('v_max:', v_eq_max)
        print('-----------------------')

        # restart the sumo instance
        self.restart_simulation(
            sim_params=self.sim_params,
            render=self.sim_params.render)

        # perform the generic reset function
        return super().reset()

class MultiAgentWaveAttenuationPONcomEnv(MultiEnv):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

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
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1,),
            dtype=np.float32)

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

    def _apply_rl_actions(self, rl_actions):
        """Split the accelerations by ring."""
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            accel = list(rl_actions.values())
            self.k.vehicle.apply_acceleration(rl_ids, accel)

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
