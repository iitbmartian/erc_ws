import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import time 
import os 
import numpy as np 
from tf2_ros import Buffer, TransformListener
try:
    from tf2_geometry_msgs import do_transform_point  # Registers geometry types for tf2 and provides helpers
    _HAS_TF2_GEOMETRY = True
except Exception:
    _HAS_TF2_GEOMETRY = False
from geometry_msgs.msg import PointStamped
from scipy.spatial.transform import Rotation
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosgraph_msgs.msg import Clock
from rclpy.qos import QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, QoSLivelinessPolicy
from rclpy.exceptions import ParameterAlreadyDeclaredException
from rclpy.time import Time

from yolo_final import YOLODetection
import matplotlib
matplotlib.use('TkAgg')  # ✅ Use TkAgg backend instead of Qt5Agg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import threading

class YoloLocalization(Node):
    def __init__(self):
        super().__init__('yolo_localization')
        self.bridge = CvBridge()
        self.detector = YOLODetection()

        # Use sim time like ArUco (do not redeclare if already set)
        try:
            self.declare_parameter('use_sim_time', True)
        except ParameterAlreadyDeclaredException:
            pass
        self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value

        # Consistent world frame for markers and transforms
        self.target_frame = 'panther/odom'
        
        self.latest_clock = None
        if self.use_sim_time:
            self.clock_sub = self.create_subscription(
                Clock,
                '/clock',
                self.clock_callback,
                QoSProfile(
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
                    durability=QoSDurabilityPolicy.VOLATILE,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=10,
                )
            )

        self.masks_data=[]
        self.classes_detected = []
        self.last_printed=set()
        self.last_masks=[]
        self.i=0

        # 10 FPS RATE LIMITING
        self.PROCESSING_FPS = 10.0
        self.PROCESSING_INTERVAL = 1.0 / self.PROCESSING_FPS
        self.last_processing_time = {}  # ✅ Per-camera timing

        # SPATIAL CLUSTERING PARAMETERS
        self.SPATIAL_MERGE_DISTANCE = 1.0  # Minimum distance between objects
        self.MAX_DETECTION_DISTANCE = 5.0  # Maximum valid detection distance
        self.MIN_DETECTIONS_FOR_FINALIZATION = 1  # Detections needed to finalize object
        self.UPDATE_INTERVAL = 3  # Detections between position updates
        self.MAX_UPDATES = 1  # After this many updates, cleanup histories (like ArUco relocations)
        self.MAX_HISTORY = 30  # Cap stored history to avoid RAM growth
        
        # ✅ MULTI-CAMERA: Camera intrinsics for all 4 cameras (like ArUco)
        self.camera_intrinsics = {
            'front': {
                "K": [1164.4581058460628, 0.0, 940.0308904736822, 0.0, 1163.1139392077903, 538.5078737184984, 0.0, 0.0, 1.0],
                "D": [0.07255916236653698, -0.10707938314510618, 0.0008716897481075691, 0.0005608203338372305, 0.04991021160987643],
                'frame_id': 'panther/camera_front'
            },
            'back': {
                "K": [1164.4581058460628, 0.0, 940.0308904736822, 0.0, 1163.1139392077903, 538.5078737184984, 0.0, 0.0, 1.0],
                "D": [0.07255916236653698, -0.10707938314510618, 0.0008716897481075691, 0.0005608203338372305, 0.04991021160987643],
                'frame_id': 'panther/camera_back'
            },
            'left': {
                "K": [1164.4581058460628, 0.0, 940.0308904736822, 0.0, 1163.1139392077903, 538.5078737184984, 0.0, 0.0, 1.0],
                "D": [0.07255916236653698, -0.10707938314510618, 0.0008716897481075691, 0.0005608203338372305, 0.04991021160987643],
                'frame_id': 'panther/camera_left'
            },
            'right': {
                "K": [1164.4581058460628, 0.0, 940.0308904736822, 0.0, 1163.1139392077903, 538.5078737184984, 0.0, 0.0, 1.0],
                "D": [0.07255916236653698, -0.10707938314510618, 0.0008716897481075691, 0.0005608203338372305, 0.04991021160987643],
                'frame_id': 'panther/camera_right'
            }
        }
        
        # Known object size assumption (similar to ArUco size-based distance)
        self.OBJECT_LONG_SIDE_M = 0.30  # 30 cm longer side
        
        # TF setup for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # OBJECT TRACKING - Location-based instead of ID-based
        self.tracked_objects = {}  # Key: object_id, Value: object data
        self.finalized_objects = {}  # Key: object_id, Value: finalized position
        self.next_object_id = 1
        
        # ✅ MULTI-CAMERA: Store current frame for each camera
        self.current_frames = {
            'front': None,
            'back': None,
            'left': None,
            'right': None
        }
        
        # ✅ MULTI-CAMERA: Subscriptions for all 4 cameras (like ArUco)
        self.camera_subs = {}
        camera_topics = {
            'front': '/panther/camera_front/image_raw',
            'back': '/panther/camera_back/image_raw',
            'left': '/panther/camera_left/image_raw',
            'right': '/panther/camera_right/image_raw'
        }
        
        for camera_name, topic in camera_topics.items():
            self.camera_subs[camera_name] = self.create_subscription(
                Image, 
                topic, 
                lambda msg, cam=camera_name: self.image_callback(msg, cam), 
                10
            )
            self.last_processing_time[camera_name] = 0  # Initialize per-camera timing
        
        # Publisher for visualization markers (same topic name pattern as ArUco)
        self.marker_pub = self.create_publisher(MarkerArray, '/yolo_markers', 10)

        # Timer for periodic marker publishing (exactly like ArUco)
        self.create_timer(0.5, self.timer_callback)  # 500ms instead of 1000ms

        # Folder setup
        timestamp = str(time.asctime()).replace(" ", "_").replace(":", "-")
        self.folder_path = f"yolo_{timestamp}"
        self.images_folder = f"{self.folder_path}/images"
        self._ensure_folder(self.folder_path)

        # CSV file for object locations
        self.csv_file = f"{self.folder_path}/object_locations.csv"
        self._init_csv()
        
        # ✅ ADD: Real-time camera display setup
        self.show_camera_feeds = True  # Set to False to disable
        self.display_frames = {
            'front': None,
            'back': None,
            'left': None,
            'right': None
        }
        self.display_detections = {
            'front': [],
            'back': [],
            'left': [],
            'right': []
        }
        
        if self.show_camera_feeds:
            # ✅ Setup display in main thread but defer actual plotting
            self.display_ready = False
            self.setup_display_timer = self.create_timer(1.0, self._delayed_setup_display)

        self.get_logger().info(f'YOLO Localization started - 4 cameras, 4 FPS, {self.SPATIAL_MERGE_DISTANCE}m spatial clustering')
        self.get_logger().info(f'Camera frames: {[intrinsics["frame_id"] for intrinsics in self.camera_intrinsics.values()]}')
        self.get_logger().info(f'Publishing markers on: /yolo_markers')

    def clock_callback(self, msg):
        """Update the latest clock time from simulation"""
        self.latest_clock = msg.clock

    def _ensure_folder(self, folder_path):
        """Create folder structure."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        images_folder = os.path.join(folder_path, "images")
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

    def _init_csv(self):
        """Initialize CSV file with headers (will be overwritten by timer)."""
        pass  # CSV is now handled entirely in timer_callback like ArUco

    def estimate_distance_from_bbox(self, bbox, image_shape, camera_name):
        """Estimate object distance using known real size and bbox pixel size (camera-only).

        Uses pinhole model: Z = f * S / s, where S is real size of the chosen side and s its pixel size.
        We assume the longer real side is 0.30 m and infer the shorter side using bbox pixel ratio.
        """
        try:
            x1, y1, x2, y2 = bbox
            w_px = max(1.0, float(x2 - x1))
            h_px = max(1.0, float(y2 - y1))

            # ✅ MULTI-CAMERA: Use camera-specific intrinsics
            K = np.array(self.camera_intrinsics[camera_name]['K']).reshape(3, 3)
            fx = float(K[0, 0])
            fy = float(K[1, 1])

            # Determine which side is longer in pixels
            if w_px >= h_px:
                S_long = self.OBJECT_LONG_SIDE_M
                s_long = w_px
                # infer short side real length from pixel ratio
                S_short = S_long * (h_px / w_px)
                s_short = h_px
                Z_long = fx * S_long / s_long
                Z_short = fy * S_short / s_short if s_short > 0 else Z_long
            else:
                S_long = self.OBJECT_LONG_SIDE_M
                s_long = h_px
                S_short = S_long * (w_px / h_px)
                s_short = w_px
                Z_long = fy * S_long / s_long
                Z_short = fx * S_short / s_short if s_short > 0 else Z_long

            # Combine the two estimates (they should be close by construction)
            Z = float((Z_long + Z_short) / 2.0)
            return max(0.05, Z)
        except Exception as e:
            self.get_logger().warn(f"Distance estimation (bbox) failed for {camera_name}: {e}")
            return None

    def camera_to_world_position(self, bbox, distance, image_shape, camera_name, cam_point=None):
        """Convert detection to world coordinates using LATEST available TF (like ArUco)."""
        try:
            if cam_point is None:
                # Fallback: approximate from bbox center ray
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # ✅ MULTI-CAMERA: Use camera-specific intrinsics
                K = np.array(self.camera_intrinsics[camera_name]['K']).reshape(3, 3)
                camera_x = (center_x - K[0, 2]) * distance / K[0, 0]
                camera_y = (center_y - K[1, 2]) * distance / K[1, 1]
                camera_z = distance
                cam_point = np.array([camera_x, camera_y, camera_z], dtype=float)

            # ✅ MULTI-CAMERA: Use camera-specific frame_id
            ps = PointStamped()
            ps.header.frame_id = self.camera_intrinsics[camera_name]['frame_id']
            ps.header.stamp = self.get_clock().now().to_msg()  # Use current time
            ps.point.x, ps.point.y, ps.point.z = float(cam_point[0]), float(cam_point[1]), float(cam_point[2])

            # ✅ FIX: Use Time() for latest available transform (like ArUco)
            tf_stamped = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_intrinsics[camera_name]['frame_id'],
                Time(),  # ✅ Latest available (like ArUco does)
                timeout=Duration(seconds=0.1)  # Shorter timeout
            )

            if _HAS_TF2_GEOMETRY:
                world_point = do_transform_point(ps, tf_stamped)
                return np.array([world_point.point.x, world_point.point.y, world_point.point.z], dtype=float)
            else:
                # Manual transform fallback
                t = tf_stamped.transform
                tx, ty, tz = float(t.translation.x), float(t.translation.y), float(t.translation.z)
                qx, qy, qz, qw = float(t.rotation.x), float(t.rotation.y), float(t.rotation.z), float(t.rotation.w)
                R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                v = np.array([ps.point.x, ps.point.y, ps.point.z], dtype=float)
                v_world = R.dot(v) + np.array([tx, ty, tz], dtype=float)
                return v_world

        except Exception as e:
            self.get_logger().warn(f"Camera->world transform failed for {camera_name}: {e}")
            return None

    def find_nearest_object(self, world_position, class_name):
        """Find nearest existing object (BOTH tracked AND finalized) within spatial merge distance."""
        if world_position is None:
            return None

        best_id = None
        best_d = 1e9

        # ✅ FIX: Check BOTH tracked AND finalized objects
        all_objects = {}
        
        # Add tracked objects (currently being accumulated)
        for obj_id, obj_data in self.tracked_objects.items():
            if len(obj_data['positions']) > 0:
                all_objects[obj_id] = obj_data['positions'][-1]  # Use latest position
        
        # Add finalized objects (completed objects)
        for obj_id, obj_data in self.finalized_objects.items():
            if obj_data.get('update_count', 0) < self.MAX_UPDATES + 1:
                all_objects[obj_id] = obj_data['position']

        self.get_logger().info(f"🔍 Checking {class_name} at {world_position} against {len(all_objects)} objects (tracked+finalized)")

        for obj_id, existing_pos in all_objects.items():
            existing_pos = np.array(existing_pos)
            d = float(np.linalg.norm(world_position - existing_pos))
            
            self.get_logger().info(f"   Object {obj_id}: distance = {d:.3f}m (threshold: {self.SPATIAL_MERGE_DISTANCE}m)")
            
            if d < self.SPATIAL_MERGE_DISTANCE and d < best_d:
                best_d = d
                best_id = obj_id

        if best_id is not None:
            self.get_logger().info(f"✅ MERGING: {class_name} into Object {best_id} (distance: {best_d:.3f}m)")
        else:
            self.get_logger().info(f"🆕 NEW OBJECT: {class_name} - No existing object within {self.SPATIAL_MERGE_DISTANCE}m")

        return best_id

    def timer_callback(self):
        """
        ROS2 timer callback for periodic marker publishing (exactly like ArUco).
        """
        # Update CSV file
        with open(self.csv_file, 'w') as f:
            f.write("object_id,class_name,x,y,z,detection_count,confidence\n")
            for object_id, obj_data in self.finalized_objects.items():
                pos = obj_data['position']
                f.write(f"{object_id},{obj_data['class_name']},{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f},{obj_data['detection_count']},0.8\n")
                print("Saved YOLO Object ID:", object_id, "Position:", pos)

        self.publish_markers()

    def publish_markers(self):
        """
        Publish visualization markers showing camera info.
        Creates sphere markers colored by class and text labels with info.
        """
        marker_array = MarkerArray()
        
        for object_id, obj_data in self.finalized_objects.items():
            # Main sphere marker (exactly like ArUco)
            marker = Marker()
            marker.header.frame_id = self.target_frame
            if self.use_sim_time and self.latest_clock is not None:
                marker.header.stamp = self.latest_clock
            else:
                marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'yolo_objects'
            marker.id = int(object_id)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            pos = obj_data['position']
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = marker.scale.y = marker.scale.z = 0.4
            
            # Keep your RED color preference
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            marker_array.markers.append(marker)
            
            # Text label showing cameras that detected this object
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = 'yolo_labels'
            text_marker.id = int(object_id) + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = marker.pose.position.x
            text_marker.pose.position.y = marker.pose.position.y
            text_marker.pose.position.z = marker.pose.position.z + 0.5
            text_marker.pose.orientation.w = 1.0
            
            text_marker.scale.z = 0.3
            text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            # ✅ Show cameras that detected this object
            cameras_seen = obj_data.get('cameras', ['unknown'])
            unique_cameras = list(set(cameras_seen))
            cameras_str = '+'.join(unique_cameras)
            
            text_marker.text = f"ID:{object_id}\n{obj_data['class_name']}\nCams:{cameras_str}"
            
            marker_array.markers.append(text_marker)
        
        self.marker_pub.publish(marker_array)

    def cropped_frame(self, x1, x2, y1, y2, frame, padding=20):
        """Crop frame with bounds checking."""
        h, w = frame.shape[:2]
        x1_new = max(0, x1-padding)
        x2_new = min(w, x2+padding)
        y1_new = max(0, y1-padding)
        y2_new = min(h, y2+padding)
        return frame[y1_new:y2_new, x1_new:x2_new]

    def _delayed_setup_display(self):
        """Setup display after ROS2 initialization is complete."""
        if not self.display_ready:
            try:
                self._setup_camera_display()
                self.display_ready = True
                self.setup_display_timer.destroy()  # Stop the setup timer
                # Create update timer instead
                self.display_timer = self.create_timer(0.1, self._update_camera_display)  # 10 FPS
                self.get_logger().info("✅ Camera display setup complete")
            except Exception as e:
                self.get_logger().error(f"❌ Failed to setup camera display: {e}")
                self.show_camera_feeds = False

    def _setup_camera_display(self):
        """Setup matplotlib display for 4 camera feeds (main thread only)."""
        try:
            plt.ion()  # Interactive mode
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Arrange cameras: front top-left, back bottom-left, left top-right, right bottom-right
            self.camera_positions = {
                'front': (0, 0),   # Top-left
                'left': (0, 1),    # Top-right
                'back': (1, 0),    # Bottom-left  
                'right': (1, 1)    # Bottom-right
            }
            
            # Initialize empty plots
            self.image_plots = {}
            for camera_name, (row, col) in self.camera_positions.items():
                ax = self.axes[row, col]
                ax.set_title(f'{camera_name.upper()} Camera', fontsize=12, fontweight='bold')
                ax.axis('off')
                # Create empty image placeholder
                self.image_plots[camera_name] = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
            
            plt.tight_layout()
            plt.show(block=False)
            
        except Exception as e:
            self.get_logger().error(f"matplotlib setup failed: {e}")
            raise

    def _update_camera_display(self):
        """Update all 4 camera feeds with bounding boxes (main thread timer)."""
        if not self.display_ready or not hasattr(self, 'fig'):
            return
            
        try:
            frames_available = 0
            
            for camera_name, (row, col) in self.camera_positions.items():
                frame = self.display_frames.get(camera_name)
                detections = self.display_detections.get(camera_name, [])
                
                if frame is not None:
                    frames_available += 1
                    # Create a copy to draw on
                    display_frame = frame.copy()
                    
                    # Draw green bounding boxes
                    for detection in detections:
                        class_name, bbox = detection
                        x1, y1, x2, y2 = bbox
                        
                        # Draw green rectangle
                        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                    (0, 255, 0), 2)  # Green color, thickness 2
                        
                        # Draw class label with background
                        label = f"{class_name}"
                        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        
                        # Background rectangle for text
                        cv2.rectangle(display_frame, (int(x1), int(y1) - label_h - 10), 
                                    (int(x1) + label_w, int(y1)), (0, 255, 0), -1)
                        
                        # White text
                        cv2.putText(display_frame, label, (int(x1), int(y1) - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Convert BGR to RGB for matplotlib
                    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
                    # Update the plot
                    self.image_plots[camera_name].set_array(display_frame_rgb)
                    
                    # Update title with detection count
                    ax = self.axes[row, col]
                    detection_count = len(detections)
                    ax.set_title(f'{camera_name.upper()} Camera ({detection_count} objects)', 
                               fontsize=12, fontweight='bold')
                else:
                    # Show placeholder for missing frames
                    ax = self.axes[row, col]
                    ax.set_title(f'{camera_name.upper()} Camera (no frame)', 
                               fontsize=12, fontweight='bold', color='red')
            
            # Force redraw
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            self.get_logger().warn(f"Display update failed: {e}")

    def image_callback(self, msg, camera_name):
        """Main image processing callback for a specific camera."""
        current_time = time.time()
        if current_time - self.last_processing_time[camera_name] < self.PROCESSING_INTERVAL:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_frames[camera_name] = frame.copy()
            
            # ✅ ADD: Update display frame
            if self.show_camera_feeds:
                self.display_frames[camera_name] = frame.copy()
                
        except Exception as e:
            self.get_logger().error(f"CV Bridge error for {camera_name}: {e}")
            return

        result = self.detector.detection(frame)
        image_shape = frame.shape[:2]

        # ✅ ADD: Store detections for display
        current_detections = []

        if result:
            for detection in result:
                class_name, class_index, bbox, masks = detection
                
                # ✅ ADD: Store detection for display
                if self.show_camera_feeds:
                    current_detections.append((class_name, bbox))
                
                # ...existing detection processing code...
                if(len(self.classes_detected)>=10 and class_name in self.classes_detected[-10:]): 
                        continue 
                self.classes_detected.append(class_name)

                percentage_threshold=0.25
                if(len(self.masks_data)>10):
                    prev_masks=self.masks_data[-10:]
                    for t in range(0,5): 
                        ar=np.sum(prev_masks[t])
                        if(np.sum(masks)in range(int(ar-(ar*percentage_threshold)),int(ar+(ar*percentage_threshold)))):
                            continue
                self.masks_data.append(masks)

                if(class_name in self.last_printed):
                    continue 
                if(len(self.last_printed)==6):
                    self.last_printed.clear() 

                for t_new in range(0,len(self.last_masks)): 
                    ar_new=np.sum(self.last_masks[t_new])
                    if(np.sum(masks)in range(int(ar_new-(ar_new*percentage_threshold)),int(ar_new+(ar_new*percentage_threshold)))):
                        continue 
                
                if(len(self.last_masks)==6):
                    self.last_masks.clear()

                # ✅ MULTI-CAMERA: Pass camera_name to distance and transform functions
                distance = self.estimate_distance_from_bbox(bbox, image_shape, camera_name)
                world_position = self.camera_to_world_position(bbox, distance, image_shape, camera_name)
                
                if world_position is None:
                    self.get_logger().warn(f"Skipping {class_name} from {camera_name} - TF transform failed")
                    continue

                self.get_logger().info(f"✅ {class_name} from {camera_name} world pos: {world_position}")

                # Find nearest object (checks both tracked and finalized)
                nearest_id = self.find_nearest_object(world_position, class_name)
                
                if nearest_id is not None:
                    # Update tracked_objects
                    if nearest_id not in self.tracked_objects:
                        # Get base info for new tracking session
                        if nearest_id in self.finalized_objects:
                            existing_class = self.finalized_objects[nearest_id]['class_name']
                            existing_classes = self.finalized_objects[nearest_id].get('all_classes', [existing_class])
                        else:
                            existing_class = class_name
                            existing_classes = [class_name]
                        
                        self.tracked_objects[nearest_id] = {
                            'class_name': existing_class,
                            'all_classes': existing_classes + [class_name] if class_name not in existing_classes else existing_classes,
                            'positions': [world_position],
                            'distances': [distance],
                            'detection_count': 1,
                            'bboxes': [bbox],
                            'cameras': [camera_name],
                            'last_update_count': 0
                        }
                    else:
                        obj_data = self.tracked_objects[nearest_id]
                        obj_data['positions'].append(world_position)
                        obj_data['distances'].append(distance)
                        obj_data['detection_count'] += 1
                        obj_data['bboxes'].append(bbox)
                        
                        if camera_name not in obj_data.get('cameras', []):
                            obj_data.setdefault('cameras', []).append(camera_name)
                        
                        if class_name not in obj_data.get('all_classes', []):
                            obj_data.setdefault('all_classes', []).append(class_name)
                        
                        if len(obj_data['positions']) > self.MAX_HISTORY:
                            obj_data['positions'].pop(0)
                        if len(obj_data['distances']) > self.MAX_HISTORY:
                            obj_data['distances'].pop(0)
                        if len(obj_data['bboxes']) > self.MAX_HISTORY:
                            obj_data['bboxes'].pop(0)
                        if len(obj_data['cameras']) > self.MAX_HISTORY:
                            obj_data['cameras'].pop(0)
                    object_id = nearest_id
                else:
                    # Create completely new object
                    object_id = self.next_object_id
                    self.next_object_id += 1
                    self.tracked_objects[object_id] = {
                        'class_name': class_name,
                        'all_classes': [class_name],
                        'positions': [world_position],
                        'distances': [distance],
                        'detection_count': 1,
                        'bboxes': [bbox],
                        'cameras': [camera_name],  # ✅ Track which camera first detected this
                        'last_update_count': 0
                    }

                self._check_object_finalization(object_id)

        # ✅ ADD: Update display detections
        if self.show_camera_feeds:
            self.display_detections[camera_name] = current_detections

        self.last_processing_time[camera_name] = current_time

    def _check_object_finalization(self, object_id):
        """Check if object should be finalized based on detection count."""
        obj_data = self.tracked_objects[object_id]
        current_count = obj_data['detection_count']
        last_update_count = obj_data['last_update_count']
        
        # ✅ IMMEDIATE FINALIZATION: Finalize on first detection
        if current_count >= self.MIN_DETECTIONS_FOR_FINALIZATION:
            if object_id not in self.finalized_objects:
                self._finalize_object(object_id, "INITIAL")
            else:
                # ✅ UPDATE ON EVERY DETECTION: More responsive
                detections_since_update = current_count - last_update_count
                if detections_since_update >= self.UPDATE_INTERVAL:
                    self._finalize_object(object_id, "UPDATE")

    # ✅ REMOVE CSV updates from _finalize_object since timer handles it
    def _finalize_object(self, object_id, update_type):
        """Finalize object position."""
        obj_data = self.tracked_objects[object_id]
        
        # Calculate average position (weighted by recency)
        positions = np.array(obj_data['positions'])
        if len(positions) > 3:
            weights = np.exp(np.arange(len(positions)) * 0.2)
            weights = weights / np.sum(weights)
            avg_position = np.sum(positions * weights.reshape(-1, 1), axis=0)
        else:
            avg_position = positions[0]
        
        avg_distance = np.mean(obj_data['distances'])
        
        # Choose the most common class name from all detections
        all_classes = obj_data.get('all_classes', [obj_data['class_name']])
        class_counts = {}
        for cls in all_classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        most_common_class = max(class_counts.items(), key=lambda x: x[1])[0]
        
        # Update finalized data
        prev_update_count = self.finalized_objects.get(object_id, {}).get('update_count', 0)
        self.finalized_objects[object_id] = {
            'class_name': most_common_class,
            'all_classes': list(set(all_classes)),
            'position': avg_position,
            'distance': avg_distance,
            'detection_count': obj_data['detection_count'],
            'update_count': prev_update_count + 1
        }
        
        # Save image
        self._save_object_image(object_id, obj_data)
        
        # Update last_update_count
        obj_data['last_update_count'] = obj_data['detection_count']
        
        # Show all classes seen for this object
        classes_str = f"{most_common_class} (also: {', '.join(set(all_classes) - {most_common_class})})" if len(set(all_classes)) > 1 else most_common_class
        
        self.get_logger().info(
            f"{update_type} FINALIZED Object {object_id} ({classes_str}): "
            f"Position: ({avg_position[0]:.3f}, {avg_position[1]:.3f}, {avg_position[2]:.3f}), "
            f"Distance: {avg_distance:.2f}m, Detections: {obj_data['detection_count']}"
        )

        # ✅ FIX: DON'T delete tracked_objects here - let them accumulate for spatial merging
        # The cleanup will happen when MAX_UPDATES is reached or in _cleanup_completed_object

    def _cleanup_completed_object(self, object_id):
        """Remove heavy tracking history for an object that has reached max updates."""
        if object_id in self.tracked_objects:
            del self.tracked_objects[object_id]
            self.get_logger().info(f"🧹 CLEANED UP Object {object_id} tracking buffer")

    def _save_object_image(self, object_id, obj_data):
        """Save cropped object image using simple numeric filename."""
        if len(obj_data['bboxes']) > 0 and len(obj_data['cameras']) > 0:
            try:
                # Use the latest bbox and camera
                bbox = obj_data['bboxes'][-1]
                camera_name = obj_data['cameras'][-1]
                
                # ✅ MULTI-CAMERA: Get frame from the specific camera
                current_frame = self.current_frames.get(camera_name)
                if current_frame is None:
                    self.get_logger().warn(f"No frame available for camera {camera_name}")
                    return
                
                x1, y1, x2, y2 = bbox
                
                # Ensure valid coordinates
                h, w = current_frame.shape[:2]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                
                if x2 > x1 and y2 > y1:
                    cropped_img = current_frame[y1:y2, x1:x2]
                    if cropped_img.size > 0:
                        # ✅ SIMPLE NUMERIC FILENAME: just object_id.jpg
                        filename = f"{self.images_folder}/{object_id}.jpg"
                        cv2.imwrite(filename, cropped_img)
                        self.get_logger().info(f"Saved object image: {filename}")
                else:
                    self.get_logger().warn(f"Invalid bbox for object {object_id}: {bbox}")
            except Exception as e:
                self.get_logger().error(f"Failed to save object image {object_id}: {e}")

    def destroy_node(self):
        """Cleanup when node is destroyed."""
        self.show_camera_feeds = False  # Stop display updates
        if hasattr(self, 'display_timer'):
            self.display_timer.destroy()
        if hasattr(self, 'fig'):
            plt.close(self.fig)
        self.get_logger().info(f"YOLO Localization completed. Finalized objects: {len(self.finalized_objects)}")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloLocalization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
