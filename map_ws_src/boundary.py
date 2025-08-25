import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import random
import math

class BoundaryPublisher(Node):
    def __init__(self):
        super().__init__('boundary_publisher')
        self.publisher = self.create_publisher(Marker, 'aruco_markers', 10)
        self.marker = Marker()
        self.marker.header.frame_id = "map"  # Or "odom" or your fixed frame
        self.marker.ns = "boundary"
        self.marker.id = 0
        self.marker.type = Marker.LINE_STRIP
        self.marker.action = Marker.ADD
        self.marker.scale.x = 0.1  # Line width (meters)
        self.marker.color.a = 1.0
        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0
        self.marker.pose.orientation.w = 1.0

        self.marker.points = self.generate_random_boundary(7, 20, 30)  # 7 points, radius between 8-10m
        # Close the loop
        self.marker.points.append(self.marker.points[0])

        # Publish marker at interval
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.publish_marker)

    def generate_random_boundary(self, num_points, min_radius, max_radius):
        points = []
        angle_increment = 2 * math.pi / num_points
        for i in range(num_points):
            angle = i * angle_increment + random.uniform(-0.1, 0.1)
            radius = random.uniform(min_radius, max_radius)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            points.append(p)
        return points

    def publish_marker(self):
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.marker)

def main(args=None):
    rclpy.init(args=args)
    node = BoundaryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

