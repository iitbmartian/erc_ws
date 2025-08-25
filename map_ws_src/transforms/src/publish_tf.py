#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import json
import os

json_file_name = "panther_transforms" + ".json"

def load_transforms(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['transforms']

def make_transform(t):
    msg = TransformStamped()
    msg.header.stamp = rclpy.time.Time().to_msg()  # static transforms, time=0
    msg.header.frame_id = t['frame_id']
    msg.child_frame_id = t['child_frame_id']
    msg.transform.translation.x = t['transform']['translation']['x']
    msg.transform.translation.y = t['transform']['translation']['y']
    msg.transform.translation.z = t['transform']['translation']['z']
    msg.transform.rotation.x = t['transform']['rotation']['x']
    msg.transform.rotation.y = t['transform']['rotation']['y']
    msg.transform.rotation.z = t['transform']['rotation']['z']
    msg.transform.rotation.w = t['transform']['rotation']['w']
    return msg

def main(args=None):
    rclpy.init(args=args)
    node = Node('multi_static_broadcaster')
    broadcaster = StaticTransformBroadcaster(node)

    # Use package share directory for config path
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'config', json_file_name
    )
    config_path = os.path.abspath(config_path)

    transforms_data = load_transforms(config_path)
    transforms = [make_transform(t) for t in transforms_data]
    broadcaster.sendTransform(transforms)
    node.get_logger().info(
        f'Published {len(transforms)} static transforms from {config_path}'
    )
    # Optionally, print descriptions for debug
    for t in transforms_data:
        node.get_logger().info(
            f"Published static transform: {t.get('description', '')} ({t['frame_id']} -> {t['child_frame_id']})"
        )
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()