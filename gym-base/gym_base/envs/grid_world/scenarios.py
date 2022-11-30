from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


class ScenarioHandler:
    def __init__(self, scenario) -> None:
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
        self.target_location = np.array([8 , 8])
        self.obstacles_location = [np.array([3, 2]),
                                   np.array([4, 2]),
                                   np.array([3, 3]),
                                   np.array([4, 3]),
                                   np.array([5, 6]),
                                   np.array([6, 6]),
                                   np.array([1, 6]),
                                   np.array([1, 7]),
                                   ]

    def setup_scenario_2(self):
        self.setup_scenario_1()  # TODO

    def setup_scenario_3(self):
        self.setup_scenario_1()  # TODO

    def display(self):
        fig, ax = plt.subplots()
        xaxis = np.arange(0, self.grid_size, 1)
        yaxis = np.arange(0, self.grid_size, 1)

        plt.xticks(xaxis)
        plt.yticks(yaxis)

        ax.add_patch(Rectangle(self.robot_arm_location, 1, 1, color='blue', label='robot arm'))
        ax.add_patch(Rectangle(self.object_location, 1, 1, color='pink', label='object'))
        ax.add_patch(Rectangle(self.target_location, 1, 1, color='green', label='target loc'))
        ax.add_patch(Rectangle(self.obstacles_location[0], 1, 1, color='red', label='obstacle'))
        for obs in self.obstacles_location[1:]:
            ax.add_patch(Rectangle(obs, 1, 1, color='red'))

        plt.grid(True)
        ax.legend()
        # plt.savefig(f'scenarios/{self.scenario}.png')
        plt.show()


