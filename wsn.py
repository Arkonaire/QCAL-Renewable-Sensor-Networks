import numpy as np

from typing import List
from itertools import combinations


class Node:

    """Python class for single node in WSN."""
    def __init__(self, pos: (float, float), data_rate: float):

        """Initialization.
        Args:
            pos: (x, y) coordinate of the node.
            data_rate: Data generation rate for node.
        """
        self.x = pos[0]
        self.y = pos[1]
        self.r = data_rate


class WSN:

    """Python class for WSN representation."""
    def __init__(self, nodes: List[Node], rho: float, alpha: float, beta_1: float, beta_2: float, charge_rate: float,
                 wcv_station: (float, float), base: (float, float), max_charge: float, min_charge: float):

        """Initialization.
        Args:
            nodes: List of WSN nodes.
            rho: Power Consumption per unit data received.
            alpha: Distance dependent scaling of Power Consumption coefficient for transmitted data.
            beta_1: Base Power Consumption coefficient for transmitted data.
            beta_2: Distance dependent Power Consumption coefficient for transmitted data.
            charge_rate: Charging rate for the nodes.
            wcv_station: Location of WCV station
            base: Location of base station
            max_charge: Max battery storage for the nodes.
            min_charge: Min charge level for reliable operation.
        """
        self.nodes = nodes
        self.rho = rho
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.charge_rate = charge_rate
        self.wcv_x = wcv_station[0]
        self.wcv_y = wcv_station[1]
        self.base_x = base[0]
        self.base_y = base[1]
        self.max_charge = max_charge
        self.min_charge = min_charge
        self.power_mat = np.zeros((len(self.nodes) + 1, len(self.nodes) + 1))
        self.build_power_matrix()

    def build_power_matrix(self):

        """Evaluate power transmission coefficients."""
        x_data = np.array([self.base_x] + [node.x for node in self.nodes])
        y_data = np.array([self.base_y] + [node.y for node in self.nodes])
        for i, j in combinations(range(len(x_data)), r=2):
            dist = np.sqrt((x_data[i] - x_data[j]) ** 2 + (y_data[i] - y_data[j]) ** 2)
            coeff = self.beta_1 + self.beta_2 * (dist ** self.alpha)
            self.power_mat[i][j] = coeff
            self.power_mat[j][i] = coeff

    def add_sensor(self, node: Node):

        """Add a sensor to the WSN.
        Args:
            node: New sensor node.
        """

        # Evaluate new coefficients
        coefflist = [0] * len(self.nodes)
        for i in range(len(coefflist)):
            dist = np.sqrt((self.nodes[i].x - node.x) ** 2 + (self.nodes[i].y - node.y) ** 2)
            coefflist[i] = self.beta_1 + self.beta_2 * (dist ** self.alpha)

        # Append to WSN
        self.nodes.append(node)
        self.power_mat = np.column_stack((self.power_mat, coefflist))
        self.power_mat = np.row_stack((self.power_mat, coefflist + [0]))

    def remove_sensor(self, idx: int):

        """Remove a sensor from the WSN.
        Args:
            idx: Index of sensor to remove.
        """
        del self.nodes[idx]
        self.power_mat = np.delete(self.power_mat, idx, axis=0)
        self.power_mat = np.delete(self.power_mat, idx, axis=1)
