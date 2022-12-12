import pdb

import numpy as np
from shapely.geometry import LineString, Point

import random


class ModeHandler:
    class Mode:
        GRASP = 0
        PUSH = 1
        POKE = 2

    class Range:
        def __init__(self, scenario_num):
            self.scenario_num = scenario_num
            self.GRASP = None
            self.POKE_DROPOFF = None
            self.PUSH = None
            self.PUSH_DROPOFF = None
            self.POKE = None
            self.instantiate_range()

        def instantiate_range(self):
            if self.scenario_num == 1:
                self.GRASP = 10
                self.POKE = 4
                self.POKE_DROPOFF = np.sqrt(9)
                self.PUSH = 2
                self.PUSH_DROPOFF = np.sqrt(12)
            elif self.scenario_num == 2:
                self.GRASP = 3
                self.POKE = 2
                self.POKE_DROPOFF = np.sqrt(4)
                self.PUSH = 1
                self.PUSH_DROPOFF = np.sqrt(2)

    class Direction:
        UP = np.array([0, 1])
        DOWN = np.array([0, -1])
        RIGHT = np.array([1, 0])
        LEFT = np.array([-1, 0])

    class Reward:
        OFF_TABLE = -1000
        UNDER_TUNNEL = -1000

    def __init__(self, grid_size, scenario_num) -> None:
        self.scenario_num = scenario_num
        self.grid_size = grid_size
        self.robot_arm_location = None
        self.object_location = None
        self.target_location = None
        self.obstacles_location = None
        self.range = self.Range(self.scenario_num)

    def reset(self, robot_arm_location, object_location, target_location, obstacles_location):
        self.robot_arm_location = robot_arm_location
        self.object_location = object_location
        self.target_location = target_location
        self.obstacles_location = obstacles_location

    def move(self, start, mode, dest):
        if mode == self.Mode.GRASP:
            return self.move_by_grasp(start, dest)

        elif mode == self.Mode.POKE:
            return self.move_by_poke(start, dest)

        elif mode == self.Mode.PUSH:
            return self.move_by_push(start, dest)

    def move_by_grasp(self, start, dest):
        if not self.pos_is_in_range_for_grasp(start, dest):
            return start
        neighbours = []
        candids = self.get_move_candidates(start, dest, neighbours, check_obstacle=False)
        if len(candids) == 0:
            return start
        return candids[0]

    def move_by_poke(self, start, dest):
        if not self.pos_in_range_for_poke_push(start, dest, self.range.POKE):
            return start
        neighbours = self.get_neighbours_for_poke_push(start, dest)
        candids = self.get_move_candidates(start, dest, neighbours, check_obstacle=True)
        num_candids = len(candids)
        if num_candids == 0:
            return start
        # make probablity range dependent on distance to goal
        dist_to_goal = np.linalg.norm(dest - start)
        # 1 till dist_to_goal=np.sqrt(9), then drops off exponentially
        if dist_to_goal < self.range.POKE_DROPOFF:  # TODO: make this dependent on the scenario
            reach_goal_prob = 1.0
        else:
            reach_goal_prob = 1.0 / (1 + np.exp(-(dist_to_goal - np.sqrt(9))))
        extra_prob = (1 - reach_goal_prob) / (num_candids - 1)
        probs = [reach_goal_prob] + [extra_prob] * (num_candids - 1)
        return random.choices(candids, probs, k=1)[0]

    def move_by_push(self, start, dest):
        if not self.pos_in_range_for_poke_push(start, dest, self.range.PUSH):
            return start
        neighbours = self.get_neighbours_for_poke_push(start, dest)
        candids = self.get_move_candidates(start, dest, neighbours, check_obstacle=True)
        num_candids = len(candids)
        if num_candids == 0:
            return start
        # make probablity range dependent on distance to goal
        dist_to_goal = np.linalg.norm(dest - start)
        if dist_to_goal < self.range.PUSH_DROPOFF:  # TODO: make this dependent on the scenario
            reach_goal_prob = 1.0
        else:
            reach_goal_prob = 1.0 / (1 + np.exp(-(dist_to_goal - np.sqrt(9))))
        extra_prob = (1 - reach_goal_prob) / (num_candids - 1)
        probs = [reach_goal_prob] + [extra_prob] * (num_candids - 1)
        return random.choices(candids, probs, k=1)[0]

    def pos_in_range_for_poke_push(self, start, dest, range):
        robot_to_current = np.linalg.norm(start - self.robot_arm_location)
        current_to_dest = np.linalg.norm(start - dest)
        # check if obstacles are in the way
        if self.is_obstacle(dest):
            return False
        # check on line for obstacles
        if current_to_dest > 1:
            line = LineString([start, dest])
            for obs in self.obstacles_location:
                if line.distance(Point(obs)) < 0.1:
                    return False
        if robot_to_current <= self.range.GRASP and current_to_dest <= range:
            return True
        return False

    def pos_is_in_range_for_grasp(self, start, dest):
        robot_to_current = np.linalg.norm(start - self.robot_arm_location)
        robot_to_dest = np.linalg.norm(dest - self.robot_arm_location)
        if self.is_obstacle(dest):
            return False
        if robot_to_current <= self.range.GRASP and robot_to_dest <= self.range.GRASP:
            return True
        return False

    def get_neighbour_cells_for_grasp(self, cell):
        neighbours = []
        for dir in [self.Direction.UP, self.Direction.DOWN,
                    self.Direction.RIGHT, self.Direction.LEFT]:
            neighbour = np.clip(cell + dir, 0, self.grid_size - 1)
            neighbours.append(neighbour)
        return neighbours

    def get_neighbours_for_poke_push(self, start, dest):
        # if moving diagonally, neighbors in all directions
        directions = [self.Direction.LEFT, self.Direction.RIGHT, self.Direction.UP, self.Direction.DOWN]
        # if moving up or down only, then the neigbours are up and down
        if start[0] == dest[0]:
            directions = [self.Direction.UP, self.Direction.DOWN]
        # if moving left or right only, then the neigbours are left and right
        elif start[1] == dest[1]:
            directions = [self.Direction.LEFT, self.Direction.RIGHT]

        neighbours = []
        for dir in directions:
            neighbour = np.clip(dest + dir, 0, self.grid_size - 1)
            neighbours.append(neighbour)
        return neighbours

    def is_obstacle(self, location):
        for obs in self.obstacles_location:
            if (obs == location).all():
                return True
        return False

    def find_furthest_reachable_cell_in_the_same_direction_for_poke_push(self, start, dest, range):
        start_to_dest_line = LineString([(start[0], start[1]),
                                         (dest[0], dest[1])])
        furthest_point = start_to_dest_line.interpolate(range)
        furthest_cell = np.clip(np.around((furthest_point.x, furthest_point.y)),
                                0, self.grid_size - 1).astype(int)
        return furthest_cell

    def get_move_candidates(self, start, dest, neighbours, check_obstacle=False):
        candids = [dest]
        candids.extend(neighbours)
        # check if the neighbours or dest is an obstacle
        for i in range(len(candids)):
            if self.is_obstacle(candids[i]):
                candids[i] = start
        if check_obstacle:
            # check if the line between start and dest is blocked by an obstacle
            for i in range(len(candids)):
                line = LineString([(start[0], start[1]), (candids[i][0], candids[i][1])])
                for obs in self.obstacles_location:
                    if line.intersects(Point(obs[0], obs[1])):
                        candids[i] = start

        return candids
