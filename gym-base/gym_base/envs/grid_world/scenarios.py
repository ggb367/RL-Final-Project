import numpy as np


class ScenarioHandler:
    def __init__(self, scenario, grid_size) -> None:
        self.scenario = scenario
        self.grid_size = grid_size
        self.setup_scenario()

    def setup_scenario(self):
        if self.scenario == 1:
            self.setup_scenario_1()
        elif self.scenario == 2:
            self.setup_scenario_2()
        elif self.scenario == 3:
            self.setup_scenario_3()

    def setup_scenario_1(self):
        self.robot_arm_location = np.array([0, 0])
        self.object_location = np.array([1, 0])
        self.object_graspable = np.array([1])
        self.target_location = np.array([3, 2])
        self.obstacles_location = [np.array([2, 2]),
                                   np.array([1, 1]),
                                   np.array([0, 1])]

    def setup_scenario_2(self):
        self.setup_scenario_1()  # TODO

    def setup_scenario_3(self):
        self.setup_scenario_1()  # TODO
