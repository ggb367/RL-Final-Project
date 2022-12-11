import pdb

import numpy as np
from shapely.geometry import LineString, Point

import random

from gym_base.envs.grid_world.sim import get_contour_point, get_pose, get_ee_vel, get_ik_solution
import pybullet as pb


class SimModeHandler:
    class Mode:
        GRASP = 0
        PUSH = 1
        POKE = 2

    def __init__(self, discretize, realtime, ikea_z):
        self.discretize = discretize
        self.realtime = realtime
        self.ikea_z = ikea_z
        self.debug_id = None

    def move(self, mode, dest, target_object, robot):
        if self.debug_id is not None:
            pb.removeUserDebugItem(self.debug_id)
        robot.move_to_default_pose(using_planner=False)

        if mode == self.Mode.GRASP:
            self.move_by_grasp(dest, target_object, robot)

        elif mode == self.Mode.POKE:
            self.move_by_poke(dest, target_object, robot)

        elif mode == self.Mode.PUSH:
            self.move_by_push(dest, target_object, robot)

        
        while target_object.is_moving():
            if not self.realtime:
                pb.stepSimulation()

        new_position = np.array(target_object.get_sim_pose(euler=True).position.tolist())[:2]
        new_position = np.around(new_position / self.discretize) * self.discretize # TODO: what is this for?
        return new_position

    def move_by_grasp(self, dest, target_object, robot):
        if not self.pos_is_in_range_for_grasp(dest, target_object):
            return
        target_pose = target_object.pose
        
        initial_target_pose = target_pose.position.tolist()
        
        target_pose.position.x = dest[0]
        target_pose.position.y = dest[1]
        target_object.relocate(target_pose)
        
        end_target_pose = target_object.get_sim_pose(euler=True).position.tolist()
        
        self.debug_id = pb.addUserDebugLine(initial_target_pose, end_target_pose, lineWidth=3, lineColorRGB=[0, 0, 255])

    def move_by_poke(self, dest, target_object, robot):
        if not self.pos_is_in_range_for_poke(dest, target_object):
            return

        robot_ee_orientation = robot.get_ee_pose().orientation
        end_ee_pose_in_dest = get_pose(dest, self.ikea_z, robot_ee_orientation)
        countour_point = get_contour_point(end_ee_pose_in_dest, target_object)
        if countour_point is None:
            return

        start_ee_pose = get_pose([countour_point.x, countour_point.y], self.ikea_z, robot_ee_orientation)

        target_object_pose = target_object.get_sim_pose(euler=True).position
        target_object_xy = [target_object_pose.x, target_object_pose.y]
        end_ee_pose_in_middle_of_obj = get_pose(target_object_xy, self.ikea_z, robot_ee_orientation)

        end_ee_pose = end_ee_pose_in_middle_of_obj
        self.debug_id = pb.addUserDebugLine(start_ee_pose.position.tolist(), end_ee_pose.position.tolist(), lineWidth=3, lineColorRGB=[0, 255, 0])

        init_joint_angles = get_ik_solution(robot, start_ee_pose)
        final_joint_angles = get_ik_solution(robot, end_ee_pose)

        if init_joint_angles is None or final_joint_angles is None:
            return

        robot.move_to_joint_angles(init_joint_angles, using_planner=False)

        # TODO: why do we have two different distance calculations?
        move_distance_obj_middle_to_dest = np.linalg.norm(np.array(end_ee_pose_in_middle_of_obj.position.tolist()) -
                                                          np.array(end_ee_pose_in_dest.position.tolist()))
        velocity_mag = 0.2 + 0.8 * move_distance_obj_middle_to_dest / 1.78
        ee_vel_vec = get_ee_vel(start_ee_pose, end_ee_pose, velocity_mag)
        start_to_end_move_distance = np.linalg.norm(np.array(start_ee_pose.position.tolist()) - np.array(end_ee_pose.position.tolist()))
        time_duration = start_to_end_move_distance / np.linalg.norm(ee_vel_vec)

        robot.execute_constant_ee_velocity(ee_vel_vec, time_duration, 'push', target_object.id)


    def move_by_push(self, dest, target_object, robot):
        if not self.pos_is_in_range_for_push(dest, target_object):
            return

        robot_ee_orientation = robot.get_ee_pose().orientation
        end_ee_pose = get_pose(dest, self.ikea_z, robot_ee_orientation)
        countour_point = get_contour_point(end_ee_pose, target_object)
        if countour_point is None:
            return
 
        start_ee_pose = get_pose([countour_point.x, countour_point.y], self.ikea_z, robot_ee_orientation)
        self.debug_id = pb.addUserDebugLine(start_ee_pose.position.tolist(), end_ee_pose.position.tolist(), lineWidth=3, lineColorRGB=[255, 0, 0])

        init_joint_angles = get_ik_solution(robot, start_ee_pose)
        final_joint_angles = get_ik_solution(robot, end_ee_pose)

        if init_joint_angles is None or final_joint_angles is None:
            return

        robot.move_to_joint_angles(init_joint_angles, using_planner=False)

        move_distance = np.linalg.norm(np.array(start_ee_pose.position.tolist())[:2] - end_ee_pose.position.tolist()[:2])
        velocity_mag = 0.1 + 0.4 * move_distance / 1.78
        ee_vel_vec = get_ee_vel(target_object.get_sim_pose(euler=True), end_ee_pose, velocity_mag)
        time_duration = move_distance / np.linalg.norm(ee_vel_vec)

        robot.execute_constant_ee_velocity(ee_vel_vec, time_duration, 'push',  target_object.id)

    def target_object_is_reachable(self, target_object):
        target_object_pos = target_object.get_sim_pose(euler=True).position
        target_object_xy = [target_object_pos.x, target_object_pos.y]
        distance = np.linalg.norm(target_object_xy)
        return distance <= 0.67

    def pos_is_in_range_for_grasp(self, position, target_object):
        distance = np.linalg.norm(position)
        return distance <= 0.67 and self.target_object_is_reachable(target_object)

    def pos_is_in_range_for_poke(self, position, target_object):
        p_x, p_y = position[0], position[1]
        t_x, t_y = target_object.get_sim_pose(euler=True).position.x, target_object.get_sim_pose(euler=True).position.y
        x_distance = abs(p_x - t_x)
        y_distance = abs(p_y - t_y)
        is_horiz = x_distance <= self.discretize
        is_vert = y_distance <= self.discretize
        
        return self.target_object_is_reachable(target_object)

    def pos_is_in_range_for_push(self, position, target_object):
        p_x, p_y = position[0], position[1]
        t_x, t_y = target_object.get_sim_pose(euler=True).position.x, target_object.get_sim_pose(euler=True).position.y
        x_distance = abs(p_x - t_x)
        y_distance = abs(p_y - t_y)
        is_horiz = x_distance <= self.discretize
        is_vert = y_distance <= self.discretize

        return self.target_object_is_reachable(target_object) 