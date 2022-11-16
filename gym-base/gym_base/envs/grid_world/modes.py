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

    def __init__(self, grid_size) -> None:
        self.grid_size = grid_size

    def move(self, observation, mode, dest):
        current_pos = observation["agent"]

        if mode == self.Mode.GRASP:
            return self.move_by_grasp(current_pos, dest)

        elif mode == self.Mode.POKE:
            return self.move_by_poke(current_pos, dest)

        elif mode == self.Mode.PUSH:
            return self.move_by_poke(current_pos, dest)

    def get_neighbour_cells(self, cell):
        neighbours = []
        directions = [self.Direction.UP, self.Direction.DOWN,
                      self.Direction.LEFT, self.Direction.RIGHT]
        for dir in directions:
            neighbour = np.clip(cell + dir, 0, self.grid_size - 1)
            neighbours.append(neighbour)
        return neighbours

    def find_furthest_reachable_cell_in_the_same_direction(self, current_pos, dest):
        current_to_dest_line = LineString([(current_pos[0], current_pos[1]),
                                           (dest[0], dest[1])])
        furthest_point = current_to_dest_line.interpolate(self.Range.GRASP)
        nearest_cell = np.clip(
            np.around([(furthest_point.x, furthest_point.y)]), 0, self.grid_size - 1)
        return nearest_cell

    def move_by_grasp(self, current_pos, dest):
        if not self.pos_is_in_range_for_grasp(current_pos, dest):
            dest = self.find_furthest_reachable_cell_in_the_same_direction(
                current_pos, dest)

        neighbours = self.get_neighbour_cells(dest)
        candids = [dest]
        candids.extend(neighbours)
        return random.choices(candids, (0.88, 0.3, 0.3, 0.3, 0.3), k=1)[0]

    def move_by_poke(self, current_pos, dest):
        return np.array([1, 0])

    def move_by_push(self, current_pos, dest):
        return np.array([1, 0])

    def pos_is_in_range_for_grasp(self, current_pos, dest):
        dist = np.linalg.norm(current_pos-dest)
        if dist < self.Range.GRASP:
            return True

    def pos_is_in_range_for_poke(self, current_pos, dest):
        return True

    def pos_is_in_range_for_poke(self, current_pos, dest):
        return True
