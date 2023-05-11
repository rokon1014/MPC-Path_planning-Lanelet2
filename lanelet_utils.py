import rospy
import numpy as np
import lanelet2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon, LineString
from centerline.geometry import Centerline
from matplotlib import pyplot as plt
from geometry_msgs.msg import (
    PolygonStamped,
    Point32,
    PointStamped,
    PoseArray,
    Pose,
    PoseStamped,
)
from scipy.spatial import distance
from visualization_msgs.msg import MarkerArray, Marker


def closest_point(px, py, curx, cury):
    """find closest point in  a list of point from a point"""

    curxy = [(curx, cury)]
    pxy = np.column_stack((px, py))

    ddd = distance.cdist(curxy, pxy, "euclidean")
    min_dist = np.min(ddd)
    min_index = np.argmin(ddd)

    closest = [min_dist, min_index]
    return closest


def redistribute_vertices(geom, distance):
    """Create more points in geometry object"""

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


def convert_xy_to_pose_stamped(_x, _y, frame_id):
    """creating posestamed message from xy"""

    pt = PoseStamped()
    pt.header.frame_id = frame_id
    pt.header.stamp = rospy.Time.now()

    pt.pose.position.x = _x
    pt.pose.position.y = _y

    return pt


def create_marker_waypoint(x, y, frame_id, color):
    """marker from list of x,y points"""

    # print(size)
    marker_array = MarkerArray()
    red = 1
    green = 0
    blue = 0

    for i in range(len(x)):
        # print("i", i)
        marker = Marker()

        marker.header.frame_id = frame_id
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0

        if color == "blue":
            red = 0.0
            green = 0.0
            blue = 1.0

        if color == "green":
            red = 0.0
            green = 1.0
            blue = 0.0

        if color == "red":
            red = 1.0
            green = 0.0
            blue = 0.0

        # print(red, green, blue)
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.frame_locked = True
        # marker.lifetime = 0.5

        marker.id = i
        marker.pose.position.x = x[i]
        marker.pose.position.y = y[i]
        marker_array.markers.append(marker)

    return marker_array


def lanelet_projection(orgin_lat, origin_lon, filename):
    """loading lanelet map file and getting appropriate information"""

    projection = lanelet2.projection.UtmProjector(
        lanelet2.io.Origin(orgin_lat, origin_lon)
    )
    map = lanelet2.io.load(filename, projection)

    return map


def get_lane_info_from_lanlet_map(lmap):
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

    print(np.shape(linestring_left), np.shape(linestring_right))

    left_points = np.zeros((len(linestring_left), 2))
    right_points = np.zeros((len(linestring_right), 2))

    for i in range(len(linestring_left)):
        left_points[i, 0] = linestring_left[i].x
        left_points[i, 1] = linestring_left[i].y

    for i in range(len(linestring_right)):
        right_points[i, 0] = linestring_right[i].x
        right_points[i, 1] = linestring_right[i].y

    lp = LineString(left_points)
    rp = LineString(right_points)

    lp_uniformly_spaced = redistribute_vertices(lp, 1)
    rp_uniformly_spaced = redistribute_vertices(rp, 1)

    left_x, left_y = lp_uniformly_spaced.coords.xy
    right_x, right_y = rp_uniformly_spaced.coords.xy

    midx = [np.mean([left_x[i], right_x[i]]) for i in range(len(left_x) - 10)]
    midy = [np.mean([left_y[i], right_y[i]]) for i in range(len(left_x) - 10)]

    # plt.plot(lp_uniformly_spaced.coords.xy[0], lp_uniformly_spaced.coords.xy[1])
    # plt.plot(rp_uniformly_spaced.coords.xy[0], rp_uniformly_spaced.coords.xy[1])
    # plt.plot(midx, midy)
    # plt.show()

    return left_x, left_y, right_x, right_y, midx, midy


if __name__ == "__main__":
    # lanlet file name
    filename = "/home/rokon/iTrust_Autonomize_Obstacle_Avoidance/maps/lanelet2/good_test_full_road_v01.osm"
    ORIGIN_LAT = -38.19412
    ORIGIN_LON = 144.294838

    lanlet_map = lanelet_projection(ORIGIN_LAT, ORIGIN_LON, filename)

    get_lane_info_from_lanlet_map(lanlet_map)
