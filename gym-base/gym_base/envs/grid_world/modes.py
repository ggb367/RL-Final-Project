import pdb

import numpy as np
from shapely.geometry import LineString, Point

import random

from gym_base.envs.grid_world.sim import get_contour_point, get_pose, get_ee_vel
import pybullet as pb


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
            reach_goal_prob = 1.0 / (1 + np.exp(-(dist_to_goal-np.sqrt(9))))
        extra_prob = (1-reach_goal_prob) / (num_candids - 1)
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
        if dist_to_goal < self.range.PUSH_DROPOFF: # TODO: make this dependent on the scenario
            reach_goal_prob = 1.0
        else:
            reach_goal_prob = 1.0 / (1 + np.exp(-(dist_to_goal - np.sqrt(9))))
        extra_prob = (1 - reach_goal_prob) / (num_candids - 1)
        probs = [reach_goal_prob] + [extra_prob] * (num_candids - 1)
        return random.choices(candids, probs, k=1)[0]

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

class SimModeHandler:
    class Mode:
        GRASP = 0
        PUSH = 1
        POKE = 2

    def __init__(self, discritize, realtime):
        self.discritize = discritize
        self.realtime = realtime

    def pos_is_in_range_for_grasp(self, dest):
        distance = np.linalg.norm(dest)
        if distance <= 0.67:  # rough estimate of the range of the robot arm
            return True
        return False

    def move(self, mode, dest, target_object, robot):
        if mode == self.Mode.GRASP:
            return self.move_by_grasp(dest, target_object, robot)

        elif mode == self.Mode.POKE:
            return self.move_by_poke(dest, target_object, robot)

        if mode == self.Mode.PUSH:
            return self.move_by_push(dest, target_object, robot)

    def move_by_grasp(self, dest, target_object, robot):
        # move by grasp in pybullet
        # this is some trash ass code that relocates the object to the destination if it is in range
        if self.pos_is_in_range_for_grasp(dest):
            target_pose = target_object.pose
            target_pose.position.x = dest[0]
            target_pose.position.y = dest[1]
            target_object.relocate(target_pose)
            return dest
        else:
            return target_object.get_position()[:2]

    def move_by_poke(self, dest, target_object, robot):
        end_ee_pose = get_pose(target_object, dest)
        end_ee_pose.orientation = robot.get_ee_pose().orientation

        countour_point = get_contour_point(
            end_ee_pose, target_object)

        start_ee_pose = get_pose(target_object, np.array(
            [countour_point.x, countour_point.y]))
        start_ee_pose.orientation = robot.get_ee_pose().orientation
        start_ee_pose.position.z += 0.03

        end_ee_pose.position.z += 0.03

        init_joint_angles = None
        count_init = 0
        while init_joint_angles is None and count_init < 100:
            init_joint_angles = robot.get_ik_solution(start_ee_pose)
            count_init += 1
        final_joint_angles = None
        count_final = 0
        while final_joint_angles is None and count_final < 100:
            final_joint_angles = robot.get_ik_solution(end_ee_pose)
            count_final += 1
        # raise error if no solution found
        if init_joint_angles is None or final_joint_angles is None:
            return target_object.get_position()[:2]
        # init_joint_angles = robot.get_ik_solution(
        #     start_ee_pose)
        # final_joint_angles = robot.get_ik_solution(
        #     end_ee_pose)

        # init_joint_angles = [1.21146268,  1.01414402, -0.32507411, -2.25405194,  1.85247139,  2.81033492,
        #                      -0.07932621]
        # final_joint_angles = [0.24571083,  1.46706513,  0.05270085, -0.93101708, -0.0385409,   2.3629429,
        #                       1.03453555]

        robot.move_to_joint_angles(init_joint_angles, using_planner=False)

        ee_vel_vec = get_ee_vel(target_object.get_sim_pose(
            euler=False), end_ee_pose, 1.2)

        target_ee_velocity = ee_vel_vec
        time_duration = 3
        object_id = target_object.id
        robot.execute_constant_ee_velocity(target_ee_velocity, time_duration, 'poke', object_id)
        # get target_object_position
        while target_object.is_moving():
            if not self.realtime:
                pb.stepSimulation()
            pass
        new_position = np.array(target_object.get_sim_pose(euler=False).position.tolist())[:2]
        # lock new_position to the grid defined by self.discritize
        new_position = np.around(new_position / self.discritize) * self.discritize
        return new_position

    def move_by_push(self, dest, target_object, robot):
        end_ee_pose = get_pose(target_object, dest)
        end_ee_pose.orientation = robot.get_ee_pose().orientation

        countour_point = get_contour_point(
            end_ee_pose, target_object)

        start_ee_pose = get_pose(target_object, np.array(
            [countour_point.x, countour_point.y]))
        start_ee_pose.orientation = robot.get_ee_pose().orientation
        start_ee_pose.position.z += 0.03

        end_ee_pose.position.z += 0.03

        init_joint_angles = None
        count_init = 0
        while init_joint_angles is None and count_init < 100:
            init_joint_angles = robot.get_ik_solution(start_ee_pose)
            count_init += 1
        final_joint_angles = None
        count_final = 0
        while final_joint_angles is None and count_final < 100:
            final_joint_angles = robot.get_ik_solution(end_ee_pose)
            count_final += 1
        # raise error if no solution found
        if init_joint_angles is None or final_joint_angles is None:
            return target_object.get_position()[:2]
        # init_joint_angles = robot.get_ik_solution(
        #     start_ee_pose)
        # final_joint_angles = robot.get_ik_solution(
        #     end_ee_pose)

        # init_joint_angles = [1.21146268,  1.01414402, -0.32507411, -2.25405194,  1.85247139,  2.81033492,
        #                      -0.07932621]
        # final_joint_angles = [0.24571083,  1.46706513,  0.05270085, -0.93101708, -0.0385409,   2.3629429,
        #                       1.03453555]

        print("init: ", init_joint_angles)
        print("final:", final_joint_angles)

        robot.move_to_joint_angles(init_joint_angles, using_planner=False)

        ee_vel_vec = get_ee_vel(target_object.get_sim_pose(
            euler=False), end_ee_pose, 0.4)

        target_ee_velocity = ee_vel_vec
        time_duration = 3
        object_id = target_object.id
        robot.execute_constant_ee_velocity(target_ee_velocity, time_duration, 'push', object_id)
        # get target_object_position
        while target_object.is_moving():
            if not self.realtime:
                pb.stepSimulation()
            pass
        new_position = np.array(target_object.get_sim_pose(euler=False).position.tolist())[:2]
        # lock new_position to the grid defined by self.discritize
        new_position = np.around(new_position / self.discritize) * self.discritize
        return new_position

    def reset(self, robot_arm_location, object_location, target_location, obstacles_location):
        pass
