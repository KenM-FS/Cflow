from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from numpy import pi, sin, cos, linspace, sqrt, power, arccos
import numpy as np

ADDITIONAL_NET_PARAMS = {
  "length": 2000,
  "speed_limit": 100,
  "resolution": 40,
  "connection_radius": 320,
  "merge_length": 200,
}

# length of vehicles in the network, in meters
VEHICLE_LENGTH = 5

class RingHighwayNetwork(Network):
    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
          for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
              raise KeyError('Network parameter "{}" not supplied'.format(p))

          super().__init__(name, vehicles, net_params, initial_config, traffic_lights)

    def specify_nodes(self, net_params):
        length = net_params.additional_params["length"]
        r = length / (2 * pi)
        merge_length = net_params.additional_params["merge_length"]
        merge_theta = ((length/4 - merge_length) / length) * 2 * pi

        nodes = [{
          "id": "bottom",
          "x": 0,
          "y": -r
        }, {
          "id": "right",
          "x": r,
          "y": 0
        }, {
          "id": "top",
          "x": 0,
          "y": r
        }, {
          "id": "left",
          "x": -r,
          "y": 0
        }, {
          "id": "merge",
          "x": - r * cos(merge_theta),
          "y": - r * sin(merge_theta)
        }]

        return nodes

    def specify_edges(self, net_params):
        length = net_params.additional_params["length"]
        resolution = net_params.additional_params["resolution"]
        r = length / (2 * pi)
        edgelen = length / 4.

        cr = net_params.additional_params["connection_radius"]
        merge_length = net_params.additional_params["merge_length"]
        merge_theta = ((length/4 - merge_length) / length) * 2 * pi

        conn_Ox = sqrt(power(cr, 2) - power(r, 2))
        conn_theta = arccos(conn_Ox / cr)
        conn_arc_len = cr * (conn_theta + 1)
        conn_arc_begin = pi - (conn_theta + 0.05)
        conn_arc_end = pi + (merge_theta)

        merge_phi = (merge_length / length) * 2 * pi

        width = 3.2

        edges = [{
          "id": "bottom",
          "type": "edgeType",
          "from": "bottom",
          "to": "right",
          "length": edgelen,
          "shape": [
            (r * cos(t), r * sin(t))
            for t in linspace(-pi / 2, 0, resolution)
          ]
        }, {
          "id": "right",
          "type": "edgeType",
          "from": "right",
          "to": "top",
          "length": edgelen,
          "shape": [
            (r * cos(t), r * sin(t))
            for t in linspace(0, pi / 2, resolution)
          ]
        }, {
          "id": "top",
          "type": "edgeType",
          "from": "top",
          "to": "left",
          "length": edgelen,
          "shape": [
            (r * cos(t), r * sin(t))
            for t in linspace(pi / 2, pi, resolution)
          ]
        }, {
          "id": "left",
          "type": "edgeType",
          "from": "left",
          "to": "merge",
          "length": edgelen - merge_length,
          "shape": [
            (r * cos(t), r * sin(t))
            for t in linspace(pi, 3 * pi / 2 - merge_phi, resolution)
          ]
        }, {
          "id": "connect",
          "type": "connectionType",
          "from": "top",
          "to": "merge",
          "length": conn_arc_len,
          "shape": [
            (cr * cos(t) + conn_Ox - 20, cr * sin(t) - 3)
            for t in linspace(conn_arc_begin, conn_arc_end, resolution)
          ]
        }, {
          "id": "merge",
          "type": "mergeType",
          "from": "merge",
          "to": "bottom",
          "length": merge_length,
          "shape": [
            ((r-width) * cos(t), (r-width) * sin(t))
            for t in linspace(3 * pi / 2 - merge_phi, 3 * pi / 2, int(resolution * merge_phi))
          ]
        }]

        return edges

    def specify_types(self, net_params):
        lanes = 3
        speed_limit = net_params.additional_params["speed_limit"]

        types = [{
          "id": "edgeType",
          "numLanes": lanes,
          "speed": speed_limit
        }, {
          "id": "mergeType",
          "numLanes": lanes + 1,
          "speed": speed_limit
        }, {
          "id": "connectionType",
          "numLanes": 1,
          "speed": 60
        }]

        return types

    def specify_connections(self, net_params):
        connections = [{
          "from": "connect",
          "to": "merge",
          "fromLane": "0",
          "toLane": "3"
        }, {
          "from": "left",
          "to": "merge",
          "fromLane": "2",
          "toLane": "2"
        }, {
          "from": "left",
          "to": "merge",
          "fromLane": "1",
          "toLane": "1"
        }, {
          "from": "left",
          "to": "merge",
          "fromLane": "0",
          "toLane": "0"
        }, {
          "from": "merge",
          "to": "bottom",
          "fromLane": "3",
          "toLane": "2"
        }, {
          "from": "merge",
          "to": "bottom",
          "fromLane": "2",
          "toLane": "2",
          "pass": "true"
        }, {
          "from": "merge",
          "to": "bottom",
          "fromLane": "1",
          "toLane": "1"
        }, {
          "from": "merge",
          "to": "bottom",
          "fromLane": "0",
          "toLane": "0"
        }, {
          "from": "right",
          "to": "connect",
          "fromLane": "2",
          "toLane": "0"
        }, {
          "from": "right",
          "to": "top",
          "fromLane": "2",
          "toLane": "2"
        }, {
          "from": "right",
          "to": "top",
          "fromLane": "1",
          "toLane": "1"
        }, {
          "from": "right",
          "to": "top",
          "fromLane": "0",
          "toLane": "0"
        }]

        return connections

    def specify_routes(self, net_params):
        rts = {
          "bottom": [
            (["bottom", "right", "top", "left", "merge"], 0.8),
            (["bottom", "right", "connect", "merge"], 0.2)
          ],
          "right": [
            (["right", "top", "left", "merge", "bottom"], 0.8),
            (["right", "connect", "merge", "bottom"], 0.2)
          ],
          "top": [
            (["top", "left", "merge", "bottom", "right"], 0.8),
            (["top", "left", "merge", "bottom", "right", "connect"], 0.2)
          ],
          "left": [
            (["left", "merge", "bottom", "right", "top"], 0.8),
            (["left", "merge", "bottom", "right", "connect"], 0.2)
          ],
          "merge": [
            (["merge", "bottom", "right", "top", "left"], 0.8),
            (["merge", "bottom", "right", "connect"], 0.2)
          ],
          "connect": [
            (["connect", "merge", "bottom", "right"], 1.0)
          ]
        }

        return rts

    def specify_edge_starts(self):
        ring_length = self.net_params.additional_params["length"]
        junction_length = 0.1

        # edgestarts = [("bottom", 0),
        #               ("right", 0.16 * ring_length + junction_length),
        #               ("top", 0.32 * ring_length + 2 * junction_length),
        #               ("left", 0.48 * ring_length + 3 * junction_length),
        #               ("connect", 0.64 * ring_length + 4 * junction_length),
        #               ("merge", 0.80 * ring_length + 5 * junction_length)]
        edgestarts = [("bottom", 0),
                      ("right", 0),
                      ("top", 0),
                      ("left", 0),
                      ("connect", 0),
                      ("merge", 0)]

        return edgestarts

    def specify_internal_edge_starts(self):
        """See parent class."""
        ring_length = self.net_params.additional_params["length"]
        junction_length = 0.1  # length of inter-edge junctions

        edgestarts = [(":right_0", 0.16 * ring_length),
                      (":top_0", 0.32 * ring_length + junction_length),
                      (":left_0", 0.48 * ring_length + 2 * junction_length),
                      (":bottom_0", 0.64 * ring_length + 3 * junction_length),
                      (":connect_0", 0.8 *ring_length + 4 * junction_length),
                      (":merge_0", ring_length + 5 * junction_length)]

        return edgestarts
