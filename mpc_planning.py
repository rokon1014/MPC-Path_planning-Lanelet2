#!/usr/bin/env python2

""" MPC path planning with Casadi Opti stack"""


import casadi


import rospy
from geometry_msgs.msg import PoseStamped, PointStamped, Pose, PoseArray, Twist, Point32
import tf
import numpy as np
import numpy.matlib
from matplotlib import pyplot as plt

# from autoware_msgs.msg import DetectedObjectArray
from visualization_msgs.msg import MarkerArray, Marker
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg


# from tf import TransformListener
import tf2_ros
import tf2_sensor_msgs
import tf2_geometry_msgs

from lanelet_utils import (
    get_lane_info_from_lanlet_map,
    lanelet_projection,
    convert_xy_to_pose_stamped,
    closest_point,
    create_marker_waypoint,
)


class Vehicle:
    mass = 5022
    lf = 4.5
    lr = 4.5


def kinematic_model(states, control, vehicle, TIME_STEP):
    """vehicle kinematic model"""
    _x = states[0]
    _y = states[1]
    _yaw = states[2]

    _delta = control[0]
    _vx = control[1]

    _beta = casadi.atan2(vehicle.lr * casadi.tan(_delta), (vehicle.lf + vehicle.lr))

    _dx = _vx * casadi.cos(_yaw + _beta)
    _dy = _vx * casadi.sin(_yaw + _beta)
    _dyaw = (_vx * casadi.cos(_beta) * casadi.tan(_delta)) / (vehicle.lf + vehicle.lr)

    _x = _x + _dx * TIME_STEP
    _y = _y + _dy * TIME_STEP
    _yaw = _yaw + _dyaw * TIME_STEP
    _st = [_x, _y, _yaw]

    return _st


class MpcPathPlanning:
    """Path planning with MPC"""

    def __init__(self, lanlet_map):
        self.current_position = []
        self.goal_pose = []
        self.mpc_i = 0
        self.cnt = [0.1, 0.1]

        self.current_pose_sub = rospy.Subscriber(
            "/current_pose", PoseStamped, self.current_pose_callback
        )

        self.goal_pose_sub = rospy.Subscriber(
            "/goal_pose", PoseStamped, self.goal_pose_callback
        )

        self.prediction_pub = rospy.Publisher(
            "/mpc_prediction_marker", MarkerArray, queue_size=10
        )

        self.center_lanelet_pub = rospy.Publisher(
            "/lane_center_marker", MarkerArray, queue_size=10
        )

    def current_pose_callback(self, data):
        """current position of the vehicle"""
        # print(data.pose.orientation)

        transform = tf_buffer.lookup_transform("velodyne", "map", rospy.Time())
        # print("FOUND TRNSFORM")
        data = tf2_geometry_msgs.do_transform_pose(data, transform)

        quaternion = (
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w,
        )
        current_yaw = tf.transformations.euler_from_quaternion(quaternion)

        self.current_position = [
            data.pose.position.x,
            data.pose.position.y,
            current_yaw[2],
        ]

    def goal_pose_callback(self, data):
        """goal position for the local planning"""

        transform = tf_buffer.lookup_transform("velodyne", "map", rospy.Time())
        # print("FOUND TRNSFORM")
        data = tf2_geometry_msgs.do_transform_pose(data, transform)
        quaternion = (
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w,
        )

        goal_yaw = tf.transformations.euler_from_quaternion(quaternion)
        # print(data.header)
        # print(goal_yaw)

        self.goal_pose = [data.pose.position.x, data.pose.position.y, goal_yaw[2]]

    def mpc_calculation(self):
        """Main mpc calculations"""

        print("calculating", self.current_position)
        if self.current_position:
            mpc_current_state = self.current_position
            mpc_goal_pose = self.goal_pose

            # get transform from map to local velodyne frame
            transform = tf_buffer.lookup_transform("velodyne", "map", rospy.Time())

            ###########################################################
            (
                left_x,
                left_y,
                right_x,
                right_y,
                mid_x,
                mid_y,
            ) = get_lane_info_from_lanlet_map(lanlet_map)

            # transform road boundary points
            transformed_pose_array = []
            for mx, my in zip(mid_x, mid_y):
                pt_left_lane = convert_xy_to_pose_stamped(mx, my, "map")
                ptr = tf2_geometry_msgs.do_transform_pose(pt_left_lane, transform)
                transformed_pose_array.append(ptr)

            mid_x_local = np.array(
                [pt.pose.position.x for pt in transformed_pose_array]
            )
            mid_y_local = np.array(
                [pt.pose.position.y for pt in transformed_pose_array]
            )

            closest_mid = closest_point(
                mid_x_local, mid_y_local, mpc_current_state[0], mpc_current_state[1]
            )

            center_x = mid_x_local[
                closest_mid[1] : (closest_mid[1] + PREDICTION_HORIZON)
            ]

            center_y = mid_y_local[
                closest_mid[1] : (closest_mid[1] + PREDICTION_HORIZON)
            ]

            marker_msg = create_marker_waypoint(
                mid_x_local, mid_y_local, "velodyne", "green"
            )
            self.center_lanelet_pub.publish(marker_msg)

            # set inital value for the mpc opeartion using the current position
            opti.set_initial(
                states, np.tile(mpc_current_state, (PREDICTION_HORIZON + 1, 1))
            )

            # set initial value for the control
            opti.set_initial(u_dv, np.tile(self.cnt, (PREDICTION_HORIZON, 1)))

            # filling parameter values
            opti.set_value(states_current, mpc_current_state)
            opti.set_value(x_ref, self.goal_pose[0])
            opti.set_value(y_ref, self.goal_pose[1])
            opti.set_value(yaw_ref, self.goal_pose[2])

            if len(center_x) == PREDICTION_HORIZON:
                opti.set_value(road_center_x, center_x)
                opti.set_value(road_center_y, center_y)
            else:
                print("returning")
                return

            try:
                solution = opti.solve()
                print("FOUND SOLUTION")
                # print(solution.get("x_dv"))
                sol_x = solution.get("x_dv")
                sol_y = solution.get("y_dv")
                c1 = solution.value(u_dv[0, 0])
                c2 = solution.value(u_dv[0, 1])
                self.cnt = [c1, c2]

            except Exception:
                print("NO Solution")
                greedy_sol = opti.debug.value(u_dv)
                sol_x = opti.debug.value(x_dv)
                sol_y = opti.debug.value(y_dv)

                c1 = opti.debug.value(u_dv[0, 0])
                c2 = opti.debug.value(u_dv[0, 1])
                self.cnt = [c1, c2]

            marker_msg = create_marker_waypoint(sol_x, sol_y, "velodyne", "red")
            self.prediction_pub.publish(marker_msg)


if __name__ == "__main__":
    rospy.init_node("Mpc_path_planning")

    # MPC Parameters
    TIME_STEP = 0.1
    PREDICTION_HORIZON = 15
    NUM_OF_STATES = 3
    NUM_OF_CONTROLS = 2

    ## Weight values
    WEIGHT_X = 50
    WEIGHT_Y = 50
    WEIGHT_YAW = 10
    WEIGHT_ROAD = 500
    WEIGHT_V = 100
    WEIGHT_DELTA = 5000

    # vehicle params
    vehicle = Vehicle()

    # Casadi init
    opti = casadi.Opti()

    # State variables
    ## states = [x,y,yaw]
    states = opti.variable(PREDICTION_HORIZON + 1, NUM_OF_STATES)

    # decision variable; bascially states separated
    x_dv = states[:, 0]
    y_dv = states[:, 1]
    yaw_dv = states[:, 2]

    # control variables
    u_dv = opti.variable(PREDICTION_HORIZON, NUM_OF_CONTROLS)

    # control variable seperated
    # steering angle/ (OR OMEGA!!)
    delta_dv = u_dv[:, 0]

    # velocity / (OR ACCEL!!)
    v_dv = u_dv[:, 1]

    ##### PARAMETERS ####
    # references (GOAL!!)
    x_ref = opti.parameter(1)
    y_ref = opti.parameter(1)
    yaw_ref = opti.parameter(1)

    road_center_x = opti.parameter(PREDICTION_HORIZON)
    road_center_y = opti.parameter(PREDICTION_HORIZON)

    # current state (init)
    states_current = opti.parameter(NUM_OF_STATES)

    ########## CONSTRAINTS and COST###########################
    ### Initial state constraints
    opti.subject_to(x_dv[0] == states_current[0])
    opti.subject_to(y_dv[0] == states_current[1])
    opti.subject_to(yaw_dv[0] == states_current[2])

    cost = 0

    for i in range(PREDICTION_HORIZON):
        _st = kinematic_model(states[i, :], u_dv[i, :], vehicle, TIME_STEP)

        opti.subject_to(x_dv[i + 1] == _st[0])
        opti.subject_to(y_dv[i + 1] == _st[1])
        opti.subject_to(yaw_dv[i + 1] == _st[2])

        # cost
        cost_states = (
            WEIGHT_X * (x_dv[i + 1] - x_ref) ** 2
            + WEIGHT_Y * (y_dv[i + 1] - y_ref) ** 2
            + WEIGHT_YAW * (yaw_dv[i + 1] - yaw_ref) ** 2
            + WEIGHT_ROAD * (x_dv[i + 1] - road_center_x[i]) ** 2
            + WEIGHT_ROAD * (y_dv[i + 1] - road_center_y[i]) ** 2
        )
        cost = cost + cost_states

        if i < PREDICTION_HORIZON - 1:
            cost_u = (
                WEIGHT_DELTA * (delta_dv[i + 1] - delta_dv[i]) ** 2
                + WEIGHT_V * (v_dv[i + 1] - v_dv[i]) ** 2
            )
            cost = cost + cost_u

        opti.subject_to(delta_dv < 2)
        opti.subject_to(delta_dv > -2)

        opti.subject_to(v_dv < 40)
        opti.subject_to(v_dv > 0)

        ## NLP stuff

        opti.minimize(cost)

        s_opts = {
            # "ipopt.max_iter": 10000,
            "ipopt.print_level": 0,
            # "ipopt.max_cpu_time": 0.5,
            # "ipopt.print_time": 0,
        }
        # p_opts = {"expand": False}
        # t_opts = {"print_level": 2}
        opti.solver("ipopt", s_opts)

    ##################################################

    #### LANELET STUFF##########
    # lanlet file name
    filename = "/home/rokon/iTrust_Autonomize_Obstacle_Avoidance/maps/lanelet2/good_test_full_road_v01.osm"
    ORIGIN_LAT = -38.19412
    ORIGIN_LON = 144.294838

    lanlet_map = lanelet_projection(ORIGIN_LAT, ORIGIN_LON, filename)

    tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    mpc = MpcPathPlanning(lanlet_map)
    # mpc.map_calculation()

    rate = rospy.Rate(5)

    while not rospy.is_shutdown():
        mpc.mpc_calculation()
        rate.sleep()
