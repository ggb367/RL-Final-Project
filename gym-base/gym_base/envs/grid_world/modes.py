import numpy as np
from shapely.geometry import LineString

import random


class ModeHandler:
    class Mode:
        GRASP = 0
        PUSH = 1
        POKE = 2

    class Range:  # TODO
        GRASP = 3
        PUSH = 2
        POKE = 1

    class Direction:
        UP = np.array([0, 1])
        DOWN = np.array([0, -1])
        RIGHT = np.array([1, 0])
        LEFT = np.array([-1, 0])

        ALL = [UP, DOWN, RIGHT, LEFT]

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
            return self.move_by_poke(start, dest)

    def get_neighbour_cells(self, cell):
        neighbours = []
        for dir in self.Direction.ALL:
            neighbour = np.clip(cell + dir, 0, self.grid_size - 1)
            neighbours.append(neighbour)
        return neighbours

    def find_furthest_reachable_cell_in_the_same_direction(self, start, dest):
        start_to_dest_line = LineString([(start[0], start[1]),
                                         (dest[0], dest[1])])
        furthest_point = start_to_dest_line.interpolate(self.Range.GRASP)
        furthest_cell = np.clip(np.around((furthest_point.x, furthest_point.y)),
                                0, self.grid_size - 1).astype(int)
        return furthest_cell

    def get_move_candidates(self, start, dest, neighbours):
        candids = [dest]
        candids.extend(neighbours)
        for i in range(len(candids)):
            if self.is_obstacle(candids[i]):
                candids[i] = start
        return candids

    def move_by_grasp(self, start, dest):
        if not self.pos_is_in_range_for_grasp(start, dest):
            dest = self.find_furthest_reachable_cell_in_the_same_direction(
                start, dest)
        neighbours = self.get_neighbour_cells(dest)
        candids = self.get_move_candidates(start, dest, neighbours)
        # The probablity to move into the specified dest is 0.88 and 0.3 in the neighbour cells
        return random.choices(candids, (0.88, 0.3, 0.3, 0.3, 0.3), k=1)[0]

    def is_obstacle(self, location):
        for obs in self.obstacles_location:
            if (obs == location).all():
                return True
        return False

    def move_by_poke(self, current_pos, dest):
        return np.array([1, 0])

    def move_by_push(self, current_pos, dest):
        return np.array([1, 0])

    def pos_is_in_range_for_grasp(self, start, dest):
        robot_to_current = np.linalg.norm(start - self.robot_arm_location, ord=2)
        robot_to_dest = np.linalg.norm(dest - self.robot_arm_location)
        if robot_to_current <= self.Range.GRASP and robot_to_dest <= self.Range.GRASP:
            return True
        return False

    def pos_is_in_range_for_poke(self, current_pos, dest):
        return True

    def pos_is_in_range_for_poke(self, current_pos, dest):
        return True
