from ruckig import InputParameter, OutputParameter, Result, Ruckig, ControlInterface

from shapely.geometry import Polygon, LineString

from shapely import affinity

from npm_base import Point, Quaternion, Pose, convert_orientation

import pdb
import numpy as np

import matplotlib.pyplot as plt


def get_contour_point(end_ee_pose, pb_target_object):
    init_ee_pose = pb_target_object.get_sim_pose(euler=False)
    object_polygon = Polygon(
        pb_target_object.get_corner_pts()).buffer(.12)

    init_to_end = LineString([(init_ee_pose.position.x, init_ee_pose.position.y),
                              (end_ee_pose.position.x, end_ee_pose.position.y)])

    intersection_before_buffer = object_polygon.boundary.intersection(
        init_to_end)

    init_to_end_scaled = affinity.scale(init_to_end, 10000, 10000)  # if you see this, no you don't
    intersection_after_buffer = object_polygon.boundary.intersection(
        init_to_end_scaled)
    contour_points = [point for point in intersection_after_buffer]
    for point in intersection_after_buffer:
        if round(point.x, 4) == round(intersection_before_buffer.x, 4) and \
                round(point.y, 4) == round(intersection_before_buffer.y, 4):
            contour_points.remove(point)
    counter_point = contour_points[0]
    return counter_point


def get_pose(pb_object, dest):
    pose = pb_object.get_sim_pose(euler=False)
    pose.position.x = dest[0]
    pose.position.y = dest[1]
    return pose


def get_ee_vel(start_ee_pose, end_ee_pose, vel_mag):
    ee_vel_vec = [0, 0, 0, 0, 0, 0]
    direction = np.array([end_ee_pose.position.x - start_ee_pose.position.x,
                          end_ee_pose.position.y - start_ee_pose.position.y,
                          end_ee_pose.position.z - start_ee_pose.position.z])
    direction = direction / np.linalg.norm(direction)
    ee_vel_vec[0:3] = direction * vel_mag
    return ee_vel_vec


def get_ruckig_trajectory():
    pass