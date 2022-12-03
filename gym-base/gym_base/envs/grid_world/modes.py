import pdb

import numpy as np
from shapely.geometry import LineString, Point

import random


class ModeHandler:
    class Mode:
        GRASP = 0
        PUSH = 1
        POKE = 2

    # TODO: These are the maximum range an action can happen.
    # Depending where the robot arm is placed and how far
    # The start and initial are from the robot arm
    # These ranges vary. The numbers are also arbitraty for the moment
    class Range:
        GRASP = 10
        POKE = 4
        PUSH = 2

    class Direction:
        UP = np.array([0, 1])
        DOWN = np.array([0, -1])
        RIGHT = np.array([1, 0])
        LEFT = np.array([-1, 0])

    class Reward:
        OFF_TABLE = -1000
        UNDER_TUNNEL = -1000

    def __init__(self, grid_size) -> None:
        self.grid_size = grid_size

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
        # neighbours = self.get_neighbour_cells_for_grasp(dest)
        neighbours = []
        candids = self.get_move_candidates(start, dest, neighbours, check_obstacle=False)
        # return random.choices(candids, (0.96, 0.01, 0.01, 0.01, 0.01), k=1)[0]
        if len(candids) == 0:
            return start
        return candids[0]

    def move_by_poke(self, start, dest):
        # if not self.pos_in_range_for_poke_push(start, dest, self.Range.POKE):
        #     dest = self.find_furthest_reachable_cell_in_the_same_direction_for_poke_push(
        #         start, dest, self.Range.POKE)
        # #TODO: Behavior change: if the dest is not in range, then the robot will not move the object
        if not self.pos_in_range_for_poke_push(start, dest, self.Range.POKE):
            return start
        # neighbours = self.get_neighbours_for_poke_push(start, dest)
        neighbours = []
        candids = self.get_move_candidates(start, dest, neighbours, check_obstacle=True)
        # return random.choices(candids, (0.88, 0.6, 0.6), k=1)[0]
        if len(candids) == 0:
            return start
        return candids[0]

    def move_by_push(self, start, dest):
        # if not self.pos_in_range_for_poke_push(start, dest, self.Range.PUSH):
        #     dest = self.find_furthest_reachable_cell_in_the_same_direction_for_poke_push(
        #         start, dest, self.Range.PUSH)
        # # TODO: Behavior change: if the dest is not in range, then the robot will not move the object
        if not self.pos_in_range_for_poke_push(start, dest, self.Range.PUSH):
            return start
        # neighbours = self.get_neighbours_for_poke_push(start, dest)
        neighbours = []
        candids = self.get_move_candidates(start, dest, neighbours, check_obstacle=True)
        # return random.choices(candids, (0.92, 0.4, 0.4), k=1)[0]
        if len(candids) == 0:
            return start
        return candids[0]

    def pos_in_range_for_poke_push(self, start, dest, range):
        robot_to_current = np.linalg.norm(start - self.robot_arm_location)
        current_to_dest = np.linalg.norm(start - dest)
        #check if obstacles are in the way
        if self.is_obstacle(dest):
            return False
        # check on line for obstacles
        if current_to_dest > 1:
            line = LineString([start, dest])
            for obs in self.obstacles_location:
                if line.distance(Point(obs)) < 0.1:
                    return False
        if robot_to_current <= self.Range.GRASP and current_to_dest <= range:
            return True
        return False

    def pos_is_in_range_for_grasp(self, start, dest):
        robot_to_current = np.linalg.norm(start - self.robot_arm_location)
        robot_to_dest = np.linalg.norm(dest - self.robot_arm_location)
        if self.is_obstacle(dest):
            return False
        if robot_to_current <= self.Range.GRASP and robot_to_dest <= self.Range.GRASP:
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
        directions = [self.Direction.LEFT, self.Direction.RIGHT]
        if start[0] == dest[0]:
            directions = [self.Direction.UP, self.Direction.DOWN]

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
