from shapely.geometry import Polygon, LineString
from shapely import affinity
from npm_base import Point, Pose
import numpy as np
import matplotlib.pyplot as plt


def plot_countour(line1, line2, object_polygon):
    plt.plot(*line1.xy, color='green', marker='o')
    plt.plot(*line2.xy, color='red')
    plt.plot(*object_polygon.exterior.xy, color='blue')
    plt.show()


def get_contour_point(end_ee_pose_in_dest, pb_target_object):
    object_pose = pb_target_object.get_sim_pose(euler=True)
    object_to_end_ee_pos = LineString([(object_pose.position.x, object_pose.position.y),
                                       (end_ee_pose_in_dest.position.x, end_ee_pose_in_dest.position.y)])

    object_polygon = Polygon(pb_target_object.get_corner_pts()).buffer(.12)
    intersection_before_buffer = object_polygon.boundary.intersection(object_to_end_ee_pos)

    if intersection_before_buffer.is_empty:
        print("Invalid action, no intersection")
        return None

    if intersection_before_buffer.geom_type == 'MultiPoint':
        intersection_before_buffer = intersection_before_buffer[0]

    object_to_end_ee_pos_scaled = affinity.scale(object_to_end_ee_pos, 10, 10)  # if you see this, no you don't
    intersection_after_buffer = object_polygon.boundary.intersection(object_to_end_ee_pos_scaled)

    contour_points = [point for point in intersection_after_buffer]
    for point in intersection_after_buffer:
        if round(point.x, 4) == round(intersection_before_buffer.x, 4) and \
                round(point.y, 4) == round(intersection_before_buffer.y, 4):
            contour_points.remove(point)

    return contour_points[0]


def get_pose(pos_xy, z, orientation):
    pos = Pose(position=Point(pos_xy[0], pos_xy[1], z), orientation=orientation)
    return pos


def get_ee_vel(start_ee_pose, end_ee_pose, vel_mag):
    ee_vel_vec = [0, 0, 0, 0, 0, 0]
    direction = np.array([end_ee_pose.position.x - start_ee_pose.position.x,
                          end_ee_pose.position.y - start_ee_pose.position.y,
                          end_ee_pose.position.z - start_ee_pose.position.z])
    direction = direction / np.linalg.norm(direction)
    ee_vel_vec[0:3] = direction * vel_mag
    return ee_vel_vec


def get_ik_solution(robot, ee_pose):
    joint_angles = None
    count_init = 0
    while joint_angles is None and count_init < 3:
        joint_angles = robot.get_ik_solution(ee_pose)
        count_init += 1
    return joint_angles
