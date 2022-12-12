from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np


class ScenarioHandler:
    def __init__(self, scenario) -> None:
        self.grid_size = None
        self.robot_arm_location = None
        self.object_location = None
        self.initial_object_location = None
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
        self.initial_object_location = np.array([2, 0])
        self.object_graspable = np.array([1])
        self.target_location = np.array([8, 8])
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
        self.initial_object_location = np.array([2, 0])
        self.object_graspable = np.array([1])
        self.target_location = np.array([3, 3])
        self.obstacles_location = [self.robot_arm_location,
                                   np.array([3, 2]),
                                   np.array([0, 2]),
                                   ]

    def setup_scenario_3(self):
        self.grid_size = 2
        self.robot_arm_location = np.array([0, 0])
        self.object_location = np.array([1, 0])
        self.object_graspable = np.array([1])
        self.target_location = np.array([1, 1])
        self.obstacles_location = [self.robot_arm_location]

    def is_reachable(self, position):
        # TODO: make this dependent on the scenario
        if self.scenario == 1:
            if np.linalg.norm(position - self.robot_arm_location) > 10:
                return False
        elif self.scenario == 2:
            if np.linalg.norm(position - self.robot_arm_location) > 3:
                return False
        return True

    def display(self, policy=None, Q=None, save=False):
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
        if save:
            plt.savefig(f'scenarios/{self.scenario}.png')
        plt.show()

    def get_next_state(self, object_location, policy):
        # check to see if object is in the goal, if so return start location
        if np.array_equal(object_location, self.target_location):
            return self.initial_object_location
        # import pdb; pdb.set_trace()
        # get the action from the policy
        next_position = policy[object_location[0], object_location[1]]
        # clean up the action
        next_position = next_position[next_position.find("'pos'") + 7:]
        next_position = next_position[:-1]
        # convert to tuple
        next_position = tuple(map(eval, [next_position]))[0]
        return next_position

    @staticmethod
    def define_action(object_location, policy):
        action = policy[object_location[0], object_location[1]]
        # clean up the action
        action = action[action.find("'mode'") + 8:10]
        if action[0] == '0':
            return 'Grasp'
        elif action[0] == '1':
            return 'Push'
        elif action[0] == '2':
            return 'Poke'

    def prepare_animation(self, ax, object_patch, policy):
        def animate(iteration):
            # remove the old object
            object_patch.remove()
            # update the object location based on policy
            self.object_location = self.get_next_state(self.object_location, policy)
            # update only location of the object_patch
            object_patch.set_xy(self.object_location)
            # add the new object
            ax.add_patch(object_patch)
            # update title with action type and goal location
            ax.set_title(
                f'Action: {self.define_action(self.object_location, policy)} to Location: {self.get_next_state(self.object_location, policy)}')
            # if at goal change title
            if np.array_equal(self.object_location, self.target_location):
                ax.set_title(f'Object at Goal Location: {self.target_location}')
            # save the frame
            plt.savefig(f'scenarios/{self.scenario}_frame_{iteration}.png')
            return ax.patches

        return animate

    def animate(self, policy):
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
        object_patch = ax.add_patch(Rectangle(self.object_location, 1, 1, color='pink', label='object'))
        ax.add_patch(Rectangle(self.target_location, 1, 1, color='green', label='target loc'))
        ax.add_patch(Rectangle(self.obstacles_location[0], 1, 1, color='red', label='obstacle'))
        for obs in self.obstacles_location[1:]:
            ax.add_patch(Rectangle(obs, 1, 1, color='red'))
        ax.add_patch(Rectangle(self.robot_arm_location, 1, 1, color='blue', label='robot arm'))
        plt.grid(True)
        ax.legend(loc=4)
        # animate
        ani = animation.FuncAnimation(fig, self.prepare_animation(ax, object_patch, policy), repeat=True, frames=3,
                                      interval=1500)
        # plt.show(block=False)
        # save the animation as a gif
        ani.save(f'scenarios/{self.scenario}.gif', writer='imagemagick', fps=1)
