#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from builtin_interfaces.msg import Time

class TwistConverter(Node):
    def __init__(self):
        super().__init__('twist_converter')
        
        # Create publisher and subscriber
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel_nav2',  # Subscribes to remapped topic
            self.twist_callback,
            10)
            
        self.publisher = self.create_publisher(
            TwistStamped,
            '/panther/cmd_vel',  # Publishes to simulation's expected topic
            10)
            
        self.get_logger().info('Twist to TwistStamped converter initialized')

    def twist_callback(self, msg):
        # Create TwistStamped message
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = self.get_clock().now().to_msg()
        twist_stamped.header.frame_id = 'panther/base_link'
        twist_stamped.twist = msg  # Copy twist data
        self.publisher.publish(twist_stamped)

def main(args=None):
    rclpy.init(args=args)
    node = TwistConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
