from flow.networks.ring import RingNetwork
from flow.networks.ring import ADDITIONAL_NET_PARAMS
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.core.params import VehicleParams
from flow.core.params import NetParams, InitialConfig, SumoParams, EnvParams
from flow.controllers import IDMController, SimLaneChangeController, ContinuousRouter
from flow.core.experiment import Experiment

ADDITIONAL_NET_PARAMS['length'] = 2000
ADDITIONAL_NET_PARAMS['lanes'] = 3
ADDITIONAL_NET_PARAMS['speed_limit'] = 100

noises = [round(n * 0.1, 1) for n in range(0, 100, 5)]
for noise in noises:
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {"noise": noise}),
        lane_change_controller=(SimLaneChangeController, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=150
    )
    sim_params = SumoParams(
        sim_step = 0.1,
        render=False,
        emission_path='data/' + str(noise) + '/',
        overtake_right=True,
    )
    net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)
    initial_config = InitialConfig(spacing="uniform")
    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
    flow_params = dict(
        exp_tag="congestion",
        env_name=AccelEnv,
        network=RingNetwork,
        simulator="traci",
        sim=sim_params,
        env=env_params,
        net=net_params,
        veh=vehicles,
        initial=initial_config,
    )

    flow_params['env'].horizon = 3000
    exp = Experiment(flow_params)

    _ = exp.run(1, convert_to_csv=True)
