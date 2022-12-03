from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


class ScenarioHandler:
    def __init__(self, scenario) -> None:
        self.grid_size = None
        self.robot_arm_location = None
        self.object_location = None
        self.object_graspable = None
        self.target_location = None
        self.obstacles_location = None
        self.scenario = scenario
        self.setup_scenario()

    def setup_scenario(self):
        if self.scenario == 1:
            self.setup_scenario_1()
        elif self.scenario == 2:
            self.setup_scenario_2()
        elif self.scenario == 3:
            self.setup_scenario_3()

    def setup_scenario_1(self):
        self.grid_size = 10
        self.robot_arm_location = np.array([0, 0])
        self.object_location = np.array([2, 0])
        self.object_graspable = np.array([1])
        self.target_location = np.array([9 , 9])
        self.obstacles_location = [np.array([3, 2]),
                                   np.array([4, 2]),
                                   np.array([3, 3]),
                                   np.array([4, 3]),
                                   np.array([5, 6]),
                                   np.array([6, 6]),
                                   np.array([1, 6]),
                                   np.array([1, 7]),
                                   self.robot_arm_location
                                   ]

    def setup_scenario_2(self):
        self.grid_size = 4
        self.robot_arm_location = np.array([0, 0])
        self.object_location = np.array([2, 0])
        self.object_graspable = np.array([1])
        self.target_location = np.array([3 , 3])
        self.obstacles_location = [self.robot_arm_location,
                                   np.array([3, 2]),
                                   np.array([0, 2]),
                                   ]


    def setup_scenario_3(self):
        self.grid_size = 2
        self.robot_arm_location = np.array([0, 0])
        self.object_location = np.array([1, 0])
        self.object_graspable = np.array([1])
        self.target_location = np.array([1 , 1])
        self.obstacles_location = [self.robot_arm_location]


    def is_reachable(self, position):
        # not reachable if further than 3 steps away from the robot arm
        if np.linalg.norm(position - self.robot_arm_location) > 10:
            return False
        return True

    def display(self, policy=None, Q=None):
        fig, ax = plt.subplots()
        xaxis = np.arange(0, self.grid_size + 1, 1)
        yaxis = np.arange(0, self.grid_size + 1, 1)

        plt.xticks(xaxis)
        plt.yticks(yaxis)
        # shade the grid if it is reachable
        for row in range(0, self.grid_size):
            for col in range(0, self.grid_size):
                if self.is_reachable(np.array([row, col])):
                    ax.add_patch(Rectangle((col, row), 1, 1, color='blue', alpha=0.2))


        ax.add_patch(Rectangle(self.object_location, 1, 1, color='pink', label='object'))
        ax.add_patch(Rectangle(self.target_location, 1, 1, color='green', label='target loc'))
        ax.add_patch(Rectangle(self.obstacles_location[0], 1, 1, color='red', label='obstacle'))
        for obs in self.obstacles_location[1:]:
            ax.add_patch(Rectangle(obs, 1, 1, color='red'))
        ax.add_patch(Rectangle(self.robot_arm_location, 1, 1, color='blue', label='robot arm'))
        # print the optimal policy in each grid cell
        if policy is not None:
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    ax.text(col + 0.5, row + 0.2, policy[col, row], va='center', ha='center')
        if Q is not None:
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    ax.text(col + 0.5, row + 0.7, Q[col, row, -1], va='center', ha='center')
        plt.grid(True)
        ax.legend()
        # plt.savefig(f'scenarios/{self.scenario}.png')
        plt.show()


