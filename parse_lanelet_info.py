#!/usr/bin/env python2

""" Test module: loading lanelet and checking if the points is inside a lane"""

import rospy
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon, LineString
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib.path import Path
import matplotlib.pyplot as plt
from geometry_msgs.msg import (
    PolygonStamped,
    Point32,
    PointStamped,
    PoseArray,
    Pose,
    PoseStamped,
)


import lanelet2
from lanelet2.core import (
    AttributeMap,
    TrafficLight,
    Lanelet,
    LineString3d,
    Point2d,
    Point3d,
    getId,
    LaneletMap,
    BoundingBox2d,
    BasicPoint2d,
)
from lanelet2.projection import UtmProjector


def redistribute_vertices(geom, distance):
    if geom.geom_type == "LineString":
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [
                geom.interpolate(float(n) / num_vert, normalized=True)
                for n in range(num_vert + 1)
            ]
        )
    elif geom.geom_type == "MultiLineString":
        parts = [redistribute_vertices(part, distance) for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError("unhandled geometry %s", (geom.geom_type,))


def parse_lanelet_info(orgin_lat, origin_lon, filename):
    """loading lanelet map file and getting appropriate information"""

    projection = lanelet2.projection.UtmProjector(
        lanelet2.io.Origin(orgin_lat, origin_lon)
    )
    map = lanelet2.io.load(filename, projection)

    return map


def find_boundary_from_lanelet(lmap):
    """Finding road boundary from lanlet map"""
    # load the layers
    ## lanelet:
    lanelets = lmap.laneletLayer
    lanelets_sorted = []

    # take all ids in a list
    lanelets_id = [elem.id for elem in lanelets]

    # taking all lanelets in a list
    # maybe can be done in a better way
    for i in lanelets_id:
        lanelets_sorted.append(lanelets[i])

    # for chosing each lanelet
    # need to add a loop
    index = 2

    # collect linestrings for the chosen lanelet
    # leftline
    linestring_left = lanelets_sorted[index].leftBound
    linestring_right = lanelets_sorted[index].rightBound

    # collect all the points in a list for making a polygon
    # left_points_x = []
    # left_points_y = []
    left_points = []

    for pt in linestring_left:
        point_left = [pt.x, pt.y]
        left_points.append(point_left)

    lp = LineString(left_points)
    multiline_left = redistribute_vertices(lp, 1)
    left_x = multiline_left.coords.xy[0]
    left_y = multiline_left.coords.xy[1]

    right_points = []

    for pt in linestring_right:
        point_right = [pt.x, pt.y]
        right_points.append(point_right)

    lp = LineString(right_points)
    multiline_right = redistribute_vertices(lp, 1)
    right_x = multiline_right.coords.xy[0]
    right_y = multiline_right.coords.xy[1]

    return left_x, left_y, right_x, right_y


def convert_xy_to_pose_array(x, y, frame_id):
    """creating posestamed message from xy"""

    pt = PoseArray()
    pt.header.frame_id = frame_id
    pt.header.stamp = rospy.Time.now()

    for i in range(len(x)):
        _ps = Pose()
        _ps.position.x = x[i]
        _ps.position.y = y[i]
        pt.poses.append(_ps)

    return pt


def convert_xy_to_point_stamped(x, y, frame_id):
    """creating posestamed message from xy"""

    pt = PointStamped()
    pt.header.frame_id = frame_id
    pt.header.stamp = rospy.Time.now()

    pt.point.x = x
    pt.point.y = y

    return pt


def convert_xy_to_pose_stamped(_x, _y, frame_id):
    """creating posestamed message from xy"""

    pt = PoseStamped()
    pt.header.frame_id = frame_id
    pt.header.stamp = rospy.Time.now()

    pt.pose.position.x = _x
    pt.pose.position.y = _y

    return pt
