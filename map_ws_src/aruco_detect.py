import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import cv2
from packaging import version
import cv2.aruco as aruco
import numpy as np
import math
import json
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import os
from ament_index_python.packages import get_package_share_directory\

simulation = False
if simulation:
    topic_config = "sim_topics"
    panther_config = "panther_sim"
else:
    topic_config = "topic_config"
    panther_config = "panther_transforms"

if version.parse(cv2.__version__) >= version.parse("4.7.0"):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
else:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()

pkg_share = get_package_share_directory('common_config')
json_path = os.path.join(pkg_share, 'config', topic_config + '.json')
with open(json_path, 'r') as f:
    config = json.load(f)
topic_dict = {t['description']: t['topic'] for t in config['topics']}

def quaternion_to_euler(q: Quaternion):
    # Converts quaternion to Euler angles (roll, pitch, yaw) in radians
    x, y, z, w = q.x, q.y, q.z, q.w
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.declare_parameter('config_file', 'aruco_config.json')
        config_path = self.get_parameter('config_file').get_parameter_value().string_value
        self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.camera_params_map = self.config['cameras']
        self.marker_size = self.config['marker_size_m']
        self.map_frame = self.config['map_frame']
        self.active_cameras = self.config.get('active_cameras', [])

        self.bridge = CvBridge()
        self.detected_ids = set()

        # Subscribe to robot pose from odometry
        self.robot_pose = None  # (x, y, yaw)
        self.create_subscription(Odometry, topic_dict['Filtered odometry data'], self.odom_callback, 10)

        # Subscribe to camera image topics
        for cam in self.active_cameras:
            topic = self.camera_params_map[cam]['topic']
            self.create_subscription(Image, topic, lambda msg, cam=cam: self.image_callback(msg, cam), 10)

        self.marker_pub = self.create_publisher(Marker, '/boundary_marker', 10)
        self.new_points = []

        self.get_logger().info(f"ArucoDetectorNode initialized with cameras: {self.active_cameras}")

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        _, _, yaw = quaternion_to_euler(msg.pose.pose.orientation)
        self.robot_pose = (x, y, yaw)

    def image_callback(self, msg, cam):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image for {cam}: {e}')
            return
        self.detect_aruco_markers(cv_image, cam)

    def detect_aruco_markers(self, frame_bgr, cam):
        cam_cfg = self.camera_params_map[cam]
        K = np.array(cam_cfg['K'], dtype=np.float64).reshape((3, 3))
        D = np.array(cam_cfg['D'], dtype=np.float64)

        undistorted = cv2.undistort(frame_bgr, K, D, None, K)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

        corners_list, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners_list, self.marker_size, K, D)
            for idx, marker_id in enumerate(ids.flatten()):
                if marker_id in self.detected_ids:
                    self.get_logger().info(f"Skipping already detected marker ID: {marker_id}")
                    continue

                tvec = tvecs[idx][0]
                rel_x = tvec[0]  # right
                rel_y = tvec[1]  # down
                rel_z = tvec[2]  # forward

                if self.robot_pose is None:
                    self.get_logger().warn(f"Robot pose not available yet, cannot transform marker {marker_id}")
                    continue

                robot_x, robot_y, robot_theta = self.robot_pose

                # Project marker position into map frame (2D plane approximation)
                marker_map_x = robot_x + (rel_z * math.cos(robot_theta) - rel_x * math.sin(robot_theta))
                marker_map_y = robot_y + (rel_z * math.sin(robot_theta) + rel_x * math.cos(robot_theta))

                pt = Point()
                pt.x = marker_map_x
                pt.y = marker_map_y
                pt.z = 0.0  # Assuming ground plane

                self.new_points.append(pt)
                self.get_logger().info(f"Detected new marker {marker_id} at map position: x={pt.x}, y={pt.y}, z={pt.z}")

                self.detected_ids.add(marker_id)

            if len(self.new_points) > 0:
                self.publish_boundary_marker()

    def publish_boundary_marker(self):
        if not self.new_points:
            return

        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "boundary"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.points = list(self.new_points)

        # # Close the loop for visualization if appropriate
        # if len(marker.points) > 2 and (marker.points[0] != marker.points[-1]):
        #     marker.points.append(marker.points[0])

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
