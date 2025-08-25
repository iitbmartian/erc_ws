#!/usr/bin/env python3

"""
ArUco Marker Localization System
================================
Camera-only ArUco detection with continuous relocation and priority-based processing.
Detects ArUco markers from 4 cameras, transforms positions to odom frame, and
maintains finalized positions with periodic updates.

Author: ERC Unity Simulation Team
Date: August 2025
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
import cv2
from packaging import version
import cv2.aruco as aruco
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
from rclpy.qos import QoSProfile, ReliabilityPolicy,  HistoryPolicy
import tf2_geometry_msgs
from rosgraph_msgs.msg import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, QoSLivelinessPolicy
import time
import os
from sensor_msgs.msg import NavSatFix
import math
from pyproj import Transformer

class ArucoLocalization(Node):
    def __init__(self):
        super().__init__('aruco_localization')
        
        # ==================== CONFIGURABLE PARAMETERS ====================
        self.use_sim_time = True

        # ArUco Detection Parameters
        self.MARKER_SIZE = 0.15                    # Physical size of ArUco markers in meters
        self.ARUCO_DICT = cv2.aruco.DICT_4X4_50   # ArUco dictionary type
        
        # Camera Configuration
        self.ACTIVE_CAMERAS = ['front', 'back', 'left', 'right']  # Cameras to use for detection
        
        # Frame Configuration
        self.MAP_FRAME = 'panther/odom'                     # Frame for publishing markers (visualization)
        self.TRANSFORM_FRAME = 'panther/odom'      # Frame for position calculations

        # Detection Thresholds
        self.INITIAL_DETECTIONS = 2                # Detections needed for initial finalization
        self.UPDATE_INTERVAL = 2                  # Detections between relocations
        self.MAX_RELOCATIONS = 2                   # Maximum number of relocations per ArUco
        
        # Quality Control Parameters
        self.MIN_MARKER_PERIMETER = 20             # Minimum marker perimeter in pixels
        self.SIDE_RATIO_THRESHOLD = 0.3            # Max variation in marker side lengths (30%)
        self.ANGLE_THRESHOLD = 25                  # Max deviation from 90° for corners (degrees)
        self.MAX_REPROJECTION_ERROR = 2.0          # Maximum reprojection error
        self.MIN_SHARPNESS = 50                    # Minimum image sharpness around marker
        
        # Distance Validation
        self.DISTANCE_HISTORY_SIZE = 10            # Number of recent distances to track
        self.DISTANCE_OUTLIER_THRESHOLD = 0.5      # Max deviation from median distance (50%)
        self.DISTANCE_CONSISTENCY_WINDOW = 5       # Window for distance consistency check
        
        # Position-Based Weighting
        self.POSITION_PENALTY_THRESHOLD = 1.0      # Distance threshold for position penalty (meters)
        self.RECENT_POSITIONS_WINDOW = 5           # Window for recent position calculation
        self.MIN_DETECTIONS_FOR_WEIGHTING = 10     # Minimum detections to apply position weighting
        
        # Processing Optimization
        self.MAX_PROCESSING_TIME = 0.05            # Max processing time per camera (seconds)
        
        # Sequential Dependencies
        self.DEPENDENCY_RANGE = 0                  # Range for checking predecessor dependencies

        self.CRS_GPS = "EPSG:4326"      # WGS84 (GPS lat/lon)
        self.CRS_POLAND = "EPSG:2178"   # ETRS89 / Poland CS92 (meters)


        # Transformer objects (reuse for efficiency)
        self.to_xy_transformer = Transformer.from_crs(self.CRS_GPS, self.CRS_POLAND, always_xy=True)
        self.to_gps_transformer = Transformer.from_crs(self.CRS_POLAND, self.CRS_GPS, always_xy=True)

        
        # ==================== END CONFIGURABLE PARAMETERS ====================
        
        # Initialize ArUco detector
        self.aruco_dict = None
        self.aruco_params = None
        self._setup_aruco_detector()
        
        # Log configuration
        self.get_logger().info(f"ArUco Localization System Configuration:")
        self.get_logger().info(f"  Active cameras: {self.ACTIVE_CAMERAS}")
        self.get_logger().info(f"  Transform frame: {self.TRANSFORM_FRAME}, Publish frame: {self.MAP_FRAME}")
        self.get_logger().info(f"  Marker size: {self.MARKER_SIZE}m")
        self.get_logger().info(f"  Detection thresholds: Initial={self.INITIAL_DETECTIONS}, Update={self.UPDATE_INTERVAL}, Max relocations={self.MAX_RELOCATIONS}")
        self.get_logger().info(f"  Dependencies: {'ENABLED' if self.DEPENDENCY_RANGE > 0 else 'DISABLED'} (range={self.DEPENDENCY_RANGE})")
        self.get_logger().info(f"  Processing limit: {self.MAX_PROCESSING_TIME*1000:.0f}ms per camera")

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
        
        # Camera intrinsics for all 4 cameras
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
        
        # Initialize system components
        self.bridge = CvBridge()
        self.aruco_detections = {}      # Store ALL detection history (never cleared)
        self.finalized_arucos = {}      # Store finalized ArUco positions (continuously updated)
        
        # TF setup for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # ROS2 setup
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=5)
        
        # Subscribe to camera topics
        self.camera_subscribers = {}
        for camera in self.ACTIVE_CAMERAS:
            topic = f'/panther/camera_{camera}/image_raw'
            self.camera_subscribers[camera] = self.create_subscription(
                Image, topic, 
                lambda msg, cam=camera: self.image_callback(msg, cam),
                qos_profile
            )
            self.get_logger().info(f"Subscribed to {topic}")
        
        # Publisher for visualization markers
        self.marker_pub = self.create_publisher(MarkerArray, '/aruco_markers', 10)
        
        # Timer for periodic marker publishing
        self.create_timer(1.0, self.timer_callback)

        # Folder for storing ArUco detection results
        timestamp = str(time.asctime()).replace(" ", "_").replace(":", "-")
        self.folder_path = f"aruco_{timestamp}"
        self.images_folder = f"{self.folder_path}/images"
        self._ensure_aruco_detections_folder(self.folder_path) 
         

        # CSV file to store ArUco Locations..
        self.csv_file = f"{self.folder_path}/locations.csv"
        self.init_gps = None

        self.gps_subscription = self.create_subscription(
            NavSatFix, 'fix', self.gps_callback,  QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        ))




        self.get_logger().info("ArUco localization system initialized successfully")


    def gps_callback(self, msg):
        if not self.init_gps:
            self.init_gps = {
                "latitude": msg.latitude,
                "longitude": msg.longitude,
            }

    def clock_callback(self, msg):
        self.latest_clock = msg.clock
    
    def _setup_aruco_detector(self):
        """
        Initialize ArUco detector with optimized parameters.
        Sets up detector parameters for reliable marker detection.
        """
        try:
            if version.parse(cv2.__version__) >= version.parse("4.7.0"):
                self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT)
                self.aruco_params = cv2.aruco.DetectorParameters()
                # Optimized parameters for better detection
                self.aruco_params.adaptiveThreshWinSizeMin = 3
                self.aruco_params.adaptiveThreshWinSizeMax = 23
                self.aruco_params.minMarkerPerimeterRate = 0.02
                self.aruco_params.maxMarkerPerimeterRate = 5.0
                self.aruco_params.minMarkerDistanceRate = 0.03
            else:
                self.aruco_dict = cv2.aruco.Dictionary_get(self.ARUCO_DICT)
                self.aruco_params = cv2.aruco.DetectorParameters_create()
                # Same parameters for older OpenCV
                self.aruco_params.adaptiveThreshWinSizeMin = 3
                self.aruco_params.adaptiveThreshWinSizeMax = 23
                self.aruco_params.minMarkerPerimeterRate = 0.02
                self.aruco_params.maxMarkerPerimeterRate = 5.0
                self.aruco_params.minMarkerDistanceRate = 0.03
            
            self.get_logger().info(f"ArUco detector initialized (OpenCV {cv2.__version__})")
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize ArUco detector: {e}")
            raise RuntimeError(f"ArUco detector initialization failed: {e}")
    
    def image_callback(self, msg, camera):
        """
        ROS2 callback for processing camera images.
        
        Args:
            msg: ROS2 Image message
            camera: Camera name (front/back/left/right)
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.detect_aruco_markers(cv_image, camera, msg.header)
        except Exception as e:
            self.get_logger().error(f'Image conversion failed for {camera}: {e}')
    
    def detect_aruco_markers(self, image, camera, header):
        """
        Detect ArUco markers in camera image with priority-based processing.
        Processes markers in ID order (smaller IDs first) with time limits.
        
        Args:
            image: OpenCV image in RGB format
            camera: Camera name (front/back/left/right)
            header: ROS2 Header with timestamp
        """
        if self.aruco_dict is None or self.aruco_params is None:
            self.get_logger().error("ArUco detector not initialized")
            return
            
        if camera not in self.camera_intrinsics:
            self.get_logger().warn(f"No intrinsics for camera {camera}")
            return
        
        # Get camera calibration parameters
        intrinsics = self.camera_intrinsics[camera]
        K = np.array(intrinsics['K'], dtype=np.float64).reshape((3, 3))
        D = np.array(intrinsics['D'], dtype=np.float64)
        
        try:
            # Enhanced image preprocessing
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced_gray = clahe.apply(gray)
            enhanced_gray = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)
            
            # Detect ArUco markers
            corners, ids, _ = aruco.detectMarkers(enhanced_gray, self.aruco_dict, parameters=self.aruco_params)
            
            if ids is not None and len(ids) > 0:
                # Priority-based processing: Sort by ArUco ID (ascending)
                marker_data = [(marker_id, i, corners[i]) for i, marker_id in enumerate(ids.flatten())]
                marker_data.sort(key=lambda x: x[0])  # Sort by marker ID
                
                # EARLY FILTERING: Remove completed ArUcos BEFORE processing
                filtered_marker_data = []
                ignored_arucos = []
                
                for marker_id, original_idx, corner in marker_data:
                    # Check if ArUco has completed all relocations
                    if (marker_id in self.finalized_arucos and 
                        marker_id not in self.aruco_detections):
                        # Completed and cleaned up
                        completed_updates = self.finalized_arucos[marker_id].get('update_count', 0)
                        ignored_arucos.append(f"ID:{marker_id}(completed:{completed_updates})")
                        continue
                    
                    # Check if max relocations reached but not yet cleaned up
                    if marker_id in self.finalized_arucos:
                        current_update_count = self.finalized_arucos[marker_id].get('update_count', 0)
                        if current_update_count >= self.MAX_RELOCATIONS + 1:
                            ignored_arucos.append(f"ID:{marker_id}(max:{current_update_count})")
                            continue
                    
                    # ArUco is still active, keep it for processing
                    filtered_marker_data.append((marker_id, original_idx, corner))
                
                # Log ignored ArUcos
                if ignored_arucos:
                    self.get_logger().info(f"Camera {camera}: IGNORED {len(ignored_arucos)} completed ArUcos: {ignored_arucos}")
                
                # If no active ArUcos, return early
                if not filtered_marker_data:
                    self.get_logger().debug(f"Camera {camera}: No active ArUcos to process")
                    return
                
                # Update marker_data to only include active ArUcos
                marker_data = filtered_marker_data
                
                # Categorize remaining markers
                active_ids = [mid for mid, _, _ in marker_data if mid not in self.finalized_arucos]
                finalized_ids = [mid for mid, _, _ in marker_data if mid in self.finalized_arucos]
                
                # Log detection summary
                if active_ids:
                    self.get_logger().info(f"Camera {camera}: Processing {len(active_ids)} new ArUcos (priority order): {active_ids}")
                if finalized_ids:
                    self.get_logger().debug(f"Camera {camera}: Updating {len(finalized_ids)} finalized ArUcos (priority order): {finalized_ids}")
                
                # Sub-pixel corner refinement
                refined_corners = []
                for marker_id, original_idx, corner in marker_data:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
                    refined_corner = cv2.cornerSubPix(enhanced_gray, corner, (5, 5), (-1, -1), criteria)
                    refined_corners.append(refined_corner)
                
                # Pose estimation for remaining markers only
                rvecs, tvecs, reprojection_errors = aruco.estimatePoseSingleMarkers(
                    refined_corners, self.MARKER_SIZE, K, D
                )
                
                # Process markers with time limit and priority
                processed_count = 0
                start_time = self.get_clock().now()
                
                for i, (marker_id, original_idx, _) in enumerate(marker_data):
                    # Check processing time limit
                    current_time = self.get_clock().now()
                    elapsed_time = (current_time - start_time).nanoseconds / 1e9
                    
                    if elapsed_time > self.MAX_PROCESSING_TIME and processed_count > 0:
                        remaining_ids = [mid for mid, _, _ in marker_data[i:]]
                        self.get_logger().warn(
                            f"Camera {camera}: Time limit reached ({elapsed_time:.3f}s), "
                            f"processed {processed_count}/{len(marker_data)}. Skipped: {remaining_ids}"
                        )
                        break
                    
                    # Quality validation
                    marker_corners = refined_corners[i][0]
                    reprojection_error = self._extract_reprojection_error(reprojection_errors, i)
                    
                    if not self._validate_marker_quality(marker_corners, reprojection_error, enhanced_gray):
                        self.get_logger().debug(f"ArUco {marker_id} failed quality check")
                        processed_count += 1
                        continue
                    
                    # Log detection details
                    center_x = np.mean(marker_corners[:, 0])
                    center_y = np.mean(marker_corners[:, 1])
                    tvec = tvecs[i][0]
                    rvec = rvecs[i][0]
                    
                    status = "RELOCATING" if marker_id in self.finalized_arucos else "NEW"
                    self.get_logger().info(
                        f"Camera {camera}: {status} ArUco {marker_id} (priority #{processed_count + 1}) "
                        f"at ({center_x:.1f}, {center_y:.1f}), 3D: ({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f})"
                    )
                    
                    # Process the detection with image data
                    self._process_aruco_pose(
                        marker_id, camera, tvec, rvec, header, 
                        image=image,  # Pass original RGB image
                        marker_corners=marker_corners  # Pass corner coordinates
                    )
                    processed_count += 1
                
                # Log processing summary
                if processed_count < len(marker_data):
                    total_time = (self.get_clock().now() - start_time).nanoseconds / 1e9
                    self.get_logger().info(
                        f"Camera {camera}: Processed {processed_count}/{len(marker_data)} "
                        f"in {total_time:.3f}s (limit: {self.MAX_PROCESSING_TIME:.3f}s)"
                    )
                    
        except Exception as e:
            self.get_logger().error(f"ArUco detection failed: {e}")
    
    def _extract_reprojection_error(self, reprojection_errors, index):
        """
        Extract reprojection error value from OpenCV array.
        
        Args:
            reprojection_errors: Array of reprojection errors from OpenCV
            index: Index of the marker
            
        Returns:
            float: Reprojection error value or None
        """
        if reprojection_errors is None or len(reprojection_errors) <= index:
            return None
        
        error_array = reprojection_errors[index]
        if hasattr(error_array, 'flatten'):
            return float(error_array.flatten()[0])
        else:
            return float(error_array)
    
    def _validate_marker_quality(self, marker_corners, reprojection_error=None, gray_image=None):
        """
        Validate ArUco marker quality based on geometric and optical properties.
        
        Args:
            marker_corners: Array of 4 corner points
            reprojection_error: Pose estimation error
            gray_image: Grayscale image for sharpness calculation
            
        Returns:
            bool: True if marker passes quality checks
        """
        # Basic validation
        if len(marker_corners) != 4:
            return False
        
        # Size validation
        perimeter = sum(np.linalg.norm(marker_corners[(i + 1) % 4] - marker_corners[i]) for i in range(4))
        if perimeter < self.MIN_MARKER_PERIMETER:
            return False
        
        # Shape validation - check if roughly square
        side_lengths = [np.linalg.norm(marker_corners[(i + 1) % 4] - marker_corners[i]) for i in range(4)]
        avg_side = np.mean(side_lengths)
        
        for side in side_lengths:
            if abs(side - avg_side) / avg_side > self.SIDE_RATIO_THRESHOLD:
                return False
        
        # Angle validation - corners should be close to 90 degrees
        angles = []
        for i in range(4):
            p1, p2, p3 = marker_corners[i], marker_corners[(i + 1) % 4], marker_corners[(i + 2) % 4]
            v1, v2 = p1 - p2, p3 - p2
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
            angles.append(np.degrees(angle))
        
        for angle in angles:
            if abs(angle - 90) > self.ANGLE_THRESHOLD:
                return False
        
        # Reprojection error validation
        if reprojection_error is not None:
            try:
                if float(reprojection_error) > self.MAX_REPROJECTION_ERROR:
                    return False
            except (ValueError, TypeError):
                pass
        
        # Sharpness validation
        if gray_image is not None:
            sharpness = self._calculate_marker_sharpness(marker_corners, gray_image)
            if sharpness < self.MIN_SHARPNESS:
                return False
        
        return True
    
    def _calculate_marker_sharpness(self, marker_corners, gray_image):
        """
        Calculate image sharpness around the marker for quality assessment.
        
        Args:
            marker_corners: Array of 4 corner points
            gray_image: Grayscale image
            
        Returns:
            float: Sharpness value (higher is sharper)
        """
        try:
            # Create mask for marker region
            mask = np.zeros(gray_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.int32(marker_corners)], 255)
            
            # Expand region slightly
            kernel = np.ones((10, 10), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Calculate Laplacian variance as sharpness measure
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            sharpness = laplacian[mask > 0].var()
            
            return sharpness
        except:
            return 100  # Return high sharpness if calculation fails
    
    def _process_aruco_pose(self, marker_id, camera, tvec_cam, rvec_cam, header, image=None, marker_corners=None):
        """
        Process ArUco pose detection with transform, validation, and tracking.
        
        Args:
            marker_id: ArUco marker ID
            camera: Camera name
            tvec_cam: Translation vector in camera frame
            rvec_cam: Rotation vector in camera frame
            header: ROS2 Header with timestamp
            image: Original RGB image (for cropping)
            marker_corners: ArUco corner coordinates (for cropping)
        """
        # Early rejection should now be handled in detect_aruco_markers()
        # These checks are kept as safety backup
        if (marker_id in self.finalized_arucos and 
            marker_id not in self.aruco_detections):
            self.get_logger().warn(f"ArUco {marker_id} from {camera}: BACKUP REJECTION - completed all relocations")
            return
        
        if marker_id in self.finalized_arucos:
            current_update_count = self.finalized_arucos[marker_id].get('update_count', 0)
            if current_update_count >= self.MAX_RELOCATIONS + 1:
                self.get_logger().warn(f"ArUco {marker_id} from {camera}: BACKUP REJECTION - max relocations reached")
                return
        
        # Transform to odom frame for position
        camera_frame = self.camera_intrinsics[camera]['frame_id']
        position_odom = self._transform_camera_to_odom(tvec_cam, camera_frame, header)
        
        if position_odom is None:
            self.get_logger().warn(f"Transform failed for ArUco {marker_id}")
            return
        
        # Use simple camera frame distance (distance from camera to ArUco)
        distance = float(np.linalg.norm(tvec_cam))
        detection_quality = self._calculate_detection_quality(tvec_cam, rvec_cam)
        
        # Initialize tracking data if new marker
        if marker_id not in self.aruco_detections:
            self.aruco_detections[marker_id] = {
                'positions_odom': [],
                'cameras': [],
                'distances': [],
                'quality_scores': [],
                'detection_count': 0,
                'distance_history': [],
                'last_update_count': 0,
                'image_data': [],  # Store image and corners for finalization
            }
            self.get_logger().info(f"Started tracking ArUco {marker_id}")
        
        detection_data = self.aruco_detections[marker_id]
        
        # Distance outlier validation
        if len(detection_data['distance_history']) > self.DISTANCE_CONSISTENCY_WINDOW:
            recent_distances = detection_data['distance_history'][-self.DISTANCE_CONSISTENCY_WINDOW:]
            median_distance = np.median(recent_distances)
            
            distance_deviation = abs(distance - median_distance) / median_distance
            if distance_deviation > self.DISTANCE_OUTLIER_THRESHOLD:
                self.get_logger().warn(
                    f"ArUco {marker_id} distance outlier rejected: {distance:.3f}m "
                    f"vs median {median_distance:.3f}m (dev: {distance_deviation:.2f})"
                )
                return
        
        # Add detection to history
        detection_data['positions_odom'].append(position_odom.copy())
        detection_data['cameras'].append(camera)
        detection_data['distances'].append(distance)
        detection_data['distance_history'].append(distance)
        detection_data['quality_scores'].append(detection_quality)
        detection_data['detection_count'] += 1
        
        # Store image data for potential cropping
        if image is not None and marker_corners is not None:
            detection_data['image_data'].append({
                'image': image.copy(),
                'corners': marker_corners.copy(),
                'camera': camera,
                'quality': detection_quality
            })
        
        # Limit distance history size
        if len(detection_data['distance_history']) > self.DISTANCE_HISTORY_SIZE:
            detection_data['distance_history'] = detection_data['distance_history'][-self.DISTANCE_HISTORY_SIZE:]
        
        # Log detection
        status = "RELOCATING" if marker_id in self.finalized_arucos else "TRACKING"
        self.get_logger().info(
            f"ArUco {marker_id} from {camera}: {status} Detection #{detection_data['detection_count']}, "
            f"Odom pos: ({position_odom[0]:.3f}, {position_odom[1]:.3f}, {position_odom[2]:.3f}), "
            f"camera distance: {distance:.3f}m, quality: {detection_quality:.2f}"
        )
        
        # Check for finalization or update
        self._check_finalization_or_update(marker_id)
    
    def _check_finalization_or_update(self, marker_id):
        """
        Check if ArUco should be finalized or updated based on detection count.
        
        Args:
            marker_id: ArUco marker ID
        """
        detection_data = self.aruco_detections[marker_id]
        current_count = detection_data['detection_count']
        last_update_count = detection_data['last_update_count']
        
        # Initial finalization
        if current_count >= self.INITIAL_DETECTIONS and marker_id not in self.finalized_arucos:
            self._finalize_or_update_aruco(marker_id, "INITIAL FINALIZATION")
        
        # Relocation updates
        elif marker_id in self.finalized_arucos:
            detections_since_last_update = current_count - last_update_count
            if detections_since_last_update >= self.UPDATE_INTERVAL:
                self._finalize_or_update_aruco(marker_id, "RELOCATION UPDATE")
    
    def _finalize_or_update_aruco(self, marker_id, update_type):
        """
        Finalize new ArUco or update existing one with position-based weighting.
        CROPS AND SAVES IMAGE when finalized.
        
        Args:
            marker_id: ArUco marker ID
            update_type: "INITIAL FINALIZATION" or "RELOCATION UPDATE"
        """
        if marker_id not in self.aruco_detections:
            return
        
        # Dependency check for initial finalization (ONLY if DEPENDENCY_RANGE > 0)
        if update_type == "INITIAL FINALIZATION" and self.DEPENDENCY_RANGE > 0:
            required_predecessors = [marker_id - i for i in range(1, self.DEPENDENCY_RANGE + 1) if marker_id - i >= 0]
            
            if required_predecessors:
                predecessor_finalized = any(pred_id in self.finalized_arucos for pred_id in required_predecessors)
                
                if not predecessor_finalized:
                    self.get_logger().info(
                        f"ArUco {marker_id} waiting for dependency: need one of {required_predecessors} finalized first"
                    )
                    return
                else:
                    finalized_predecessors = [pid for pid in required_predecessors if pid in self.finalized_arucos]
                    self.get_logger().info(f"ArUco {marker_id} dependency satisfied: {finalized_predecessors} finalized")
            else:
                self.get_logger().info(f"ArUco {marker_id} has no dependencies (ID < {self.DEPENDENCY_RANGE})")
        elif update_type == "INITIAL FINALIZATION" and self.DEPENDENCY_RANGE == 0:
            self.get_logger().debug(f"ArUco {marker_id} dependencies disabled (DEPENDENCY_RANGE = 0)")
        
        # Check relocation limit
        elif update_type == "RELOCATION UPDATE":
            current_update_count = self.finalized_arucos.get(marker_id, {}).get('update_count', 0)
            
            if current_update_count >= self.MAX_RELOCATIONS + 1:  # +1 for initial finalization
                self.get_logger().info(f"ArUco {marker_id} reached maximum relocations ({self.MAX_RELOCATIONS})")
                return
        
        detection_data = self.aruco_detections[marker_id]
        positions_odom = np.array(detection_data['positions_odom'])
        quality_scores = np.array(detection_data['quality_scores'])
        distances = np.array(detection_data['distances'])
        
        # SAVE IMAGE when ArUco is finalized
        if update_type == "INITIAL FINALIZATION":
            image_data = detection_data.get('image_data', [])
            self._save_aruco_image(marker_id, image_data, "initial")
        elif update_type == "RELOCATION UPDATE":
            # Save recent images for relocations
            recent_images = detection_data.get('image_data', [])[-3:]  # Last 3 images
            self._save_aruco_image(marker_id, recent_images, "update")
        
        # Position-based weighting for relocations
        if update_type == "RELOCATION UPDATE" and len(positions_odom) > self.MIN_DETECTIONS_FOR_WEIGHTING:
            weights = self._calculate_position_weights(positions_odom, quality_scores, marker_id)
        else:
            weights = quality_scores / np.sum(quality_scores)
        
        # Weighted averaging
        avg_position_odom = np.sum(positions_odom * weights.reshape(-1, 1), axis=0)
        avg_distance = np.sum(distances * weights)
        avg_quality = np.mean(quality_scores)
        
        # Determine best camera
        camera_quality = {}
        for i, (cam, quality) in enumerate(zip(detection_data['cameras'], quality_scores)):
            camera_quality[cam] = camera_quality.get(cam, 0) + quality * weights[i]
        best_camera = max(camera_quality, key=camera_quality.get)
        
        # Update finalized data
        new_update_count = self.finalized_arucos.get(marker_id, {}).get('update_count', 0) + 1
        
        self.finalized_arucos[marker_id] = {
            'id': int(marker_id),
            'position_odom': avg_position_odom,
            'distance': float(avg_distance),
            'camera': best_camera,
            'detection_count': len(positions_odom),
            'quality_score': float(avg_quality),
            'update_count': new_update_count
        }
        
        # Update last_update_count BEFORE cleanup (important!)
        detection_data['last_update_count'] = detection_data['detection_count']
        
        # Log finalization/update
        camera_counts = {}
        for cam in detection_data['cameras']:
            camera_counts[cam] = camera_counts.get(cam, 0) + 1
        camera_distribution = ', '.join([f"{cam}: {count}" for cam, count in camera_counts.items()])
        
        relocations_remaining = self.MAX_RELOCATIONS + 1 - new_update_count
        
        self.get_logger().info(
            f"{update_type} ArUco {marker_id} (#{new_update_count}/{self.MAX_RELOCATIONS + 1}): "
            f"Position: ({avg_position_odom[0]:.3f}, {avg_position_odom[1]:.3f}, {avg_position_odom[2]:.3f}), "
            f"Distance: {avg_distance:.3f}m, Best camera: {best_camera}, Quality: {avg_quality:.2f}, "
            f"Detections: {len(positions_odom)} from ({camera_distribution}), "
            f"Relocations remaining: {relocations_remaining}"
        )
        
        # Clean up detection data if max relocations reached (AFTER logging)
        if new_update_count >= self.MAX_RELOCATIONS + 1:
            self._cleanup_completed_aruco(marker_id)
        
        # Check dependent ArUcos for initial finalization (ONLY if DEPENDENCY_RANGE > 0)
        if update_type == "INITIAL FINALIZATION" and self.DEPENDENCY_RANGE > 0:
            self._check_dependent_arucos(marker_id)

    def _cleanup_completed_aruco(self, marker_id):
        """
        Clean up all detection data for ArUco that has reached maximum relocations.
        Keeps only essential finalized data to prevent memory leaks.
        
        Args:
            marker_id: ArUco marker ID that reached max relocations
        """
        if marker_id in self.aruco_detections:
            # Log cleanup info
            detection_data = self.aruco_detections[marker_id]
            total_detections = detection_data['detection_count']
            total_images = len(detection_data.get('image_data', []))
            
            # Calculate approximate memory freed
            approx_memory_mb = (
                len(detection_data['positions_odom']) * 3 * 8 +  # positions (3 floats * 8 bytes)
                len(detection_data['cameras']) * 10 +  # camera strings
                len(detection_data['distances']) * 8 +  # distances (float64)
                len(detection_data['quality_scores']) * 8 +  # quality scores
                total_images * 6  # approx 6MB per image
            ) / (1024 * 1024)
            
            # Remove all detection history
            del self.aruco_detections[marker_id]
            
            self.get_logger().info(
                f"CLEANUP: ArUco {marker_id} completed all relocations. "
                f"Freed ~{approx_memory_mb:.1f}MB: {total_detections} detections, "
                f"{total_images} images. Position locked at: "
                f"({self.finalized_arucos[marker_id]['position_odom'][0]:.3f}, "
                f"{self.finalized_arucos[marker_id]['position_odom'][1]:.3f}, "
                f"{self.finalized_arucos[marker_id]['position_odom'][2]:.3f})"
            )

    def _check_dependent_arucos(self, newly_finalized_id):
        """
        Check if any waiting ArUcos can now be finalized due to dependency satisfaction.
        Only called when DEPENDENCY_RANGE > 0.
        
        Args:
            newly_finalized_id: ID of the ArUco that was just finalized
        """
        if self.DEPENDENCY_RANGE == 0:
            return  # Dependencies disabled
            
        for potential_dependent_id in range(newly_finalized_id + 1, newly_finalized_id + self.DEPENDENCY_RANGE + 1):
            if (potential_dependent_id in self.aruco_detections and 
                potential_dependent_id not in self.finalized_arucos):
                
                detection_data = self.aruco_detections[potential_dependent_id]
                if detection_data['detection_count'] >= self.INITIAL_DETECTIONS:
                    self.get_logger().info(
                        f"Checking if ArUco {potential_dependent_id} can be finalized due to ArUco {newly_finalized_id}"
                    )
                    self._check_finalization_or_update(potential_dependent_id)
    
    def _calculate_position_weights(self, positions_odom, quality_scores, marker_id):
        """
        Calculate position-based weights to reduce influence of distant old detections.
        
        Args:
            positions_odom: Array of positions in odom frame
            quality_scores: Array of detection quality scores
            marker_id: ArUco marker ID for logging
            
        Returns:
            np.array: Normalized weights for each detection
        """
        # Use recent positions as reference for current expected position
        recent_positions = positions_odom[-self.RECENT_POSITIONS_WINDOW:]
        recent_center = np.mean(recent_positions, axis=0)
        
        position_weights = []
        distance_penalties = []
        
        for i, pos in enumerate(positions_odom):
            position_distance = np.linalg.norm(pos - recent_center)
            
            if position_distance > self.POSITION_PENALTY_THRESHOLD:
                # Exponential decay for distant detections
                position_penalty = np.exp(-position_distance)
            else:
                position_penalty = 1.0  # No penalty for nearby detections
            
            position_weights.append(position_penalty)
            distance_penalties.append(position_distance)
        
        position_weights = np.array(position_weights)
        combined_weights = quality_scores * position_weights
        
        # Normalize weights
        if np.sum(combined_weights) > 0:
            weights = combined_weights / np.sum(combined_weights)
        else:
            weights = quality_scores / np.sum(quality_scores)
        
        # Log weighting info
        old_detections = np.sum(np.array(distance_penalties) > self.POSITION_PENALTY_THRESHOLD)
        avg_penalty = np.mean(distance_penalties)
        
        self.get_logger().info(
            f"ArUco {marker_id} position weighting: {old_detections}/{len(positions_odom)} "
            f"detections >{self.POSITION_PENALTY_THRESHOLD}m from recent center, avg distance: {avg_penalty:.2f}m"
        )
        
        return weights
    
    
    def _transform_camera_to_odom(self, tvec_cam, camera_frame, header):
        """
        Transform ArUco position from camera frame to odom frame using TF2.
        
        Args:
            tvec_cam: Translation vector in camera frame
            camera_frame: Camera frame ID
            header: ROS2 Header with timestamp
            
        Returns:
            np.array: Position in odom frame or None if transform fails
        """
        try:
            # Create point in camera frame
            point_stamped = PointStamped()
            point_stamped.header.frame_id = camera_frame
            point_stamped.header.stamp.sec = 0  # Use latest available transform
            point_stamped.point.x = float(tvec_cam[0])
            point_stamped.point.y = float(tvec_cam[1])
            point_stamped.point.z = float(tvec_cam[2])
            
            # Transform to odom frame
            odom_point = self.tf_buffer.transform(
                point_stamped, 
                self.TRANSFORM_FRAME,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            return np.array([odom_point.point.x, odom_point.point.y, odom_point.point.z])
            
        except Exception as e:
            self.get_logger().warn(f"Transform to {self.TRANSFORM_FRAME} failed: {e}")
            # Try fallback with current time
            try:
                if self.use_sim_time and self.latest_clock is not None:
                    point_stamped.header.stamp = self.latest_clock
                else:
                    point_stamped.header.stamp = self.get_clock().now().to_msg()
                odom_point = self.tf_buffer.transform(
                    point_stamped, 
                    self.TRANSFORM_FRAME,
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
                return np.array([odom_point.point.x, odom_point.point.y, odom_point.point.z])
            except Exception as e2:
                self.get_logger().error(f"Transform fallback failed: {e2}")
                return None
    
    def _calculate_detection_quality(self, tvec_cam, rvec_cam):
        """
        Calculate detection quality score based on distance and viewing angle.
        
        Args:
            tvec_cam: Translation vector in camera frame
            rvec_cam: Rotation vector in camera frame
            
        Returns:
            float: Quality score between 0.05 and 1.0
        """
        distance = np.linalg.norm(tvec_cam)
        
        # Distance factor (optimal range 0.8-4.0m)
        if distance < 0.8:
            distance_factor = distance * 0.7
        elif distance <= 4.0:
            distance_factor = 1.0
        elif distance <= 8.0:
            distance_factor = 0.8
        else:
            distance_factor = 4.0 / distance
        
        # Viewing angle factor
        angle_magnitude = np.linalg.norm(rvec_cam)
        angle_factor = np.cos(angle_magnitude * 1.5)
        
        # Perpendicular factor (prefer markers perpendicular to camera)
        perpendicular_factor = 1.0 - min(angle_magnitude / np.pi, 0.8)
        
        quality = distance_factor * angle_factor * perpendicular_factor
        return max(0.05, min(1.0, quality))
    
    def timer_callback(self):
        """
        ROS2 timer callback for periodic marker publishing.
        """

        with open(self.csv_file, 'w') as f:
            f.write("id,x,y,z,lat,lon\n")
            for marker_id, detection in self.finalized_arucos.items():
                pos = detection['position_odom']
                if self.init_gps:

                    x_gps, y_gps = self.gps_to_xy(self.init_gps['latitude'], self.init_gps['longitude'])
                    x_gps += pos[0]
                    y_gps += pos[1]
                    lat , lon = self.xy_to_gps(x_gps, y_gps)
                    f.write(f"{marker_id},{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f},{lat:.5f},{lon:.5f}\n")
                else:
                    lat, lon = None, None
                    f.write(f"{marker_id},{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f},N/A,N/A\n")

                print("Saved ArUco ID:", marker_id, "Position:", pos)


        self.publish_markers()
    
    def publish_markers(self):
        """
        Publish visualization markers for all finalized ArUcos.
        Creates sphere markers colored by best camera and text labels with info.
        """
        marker_array = MarkerArray()
        
        # Color mapping for cameras
        colors = {
            'front': [1.0, 0.0, 0.0],   # Red
            'back': [0.0, 1.0, 0.0],    # Green
            'left': [0.0, 0.0, 1.0],    # Blue
            'right': [1.0, 1.0, 0.0]    # Yellow
        }
        
        for marker_id, detection in self.finalized_arucos.items():
            # Main sphere marker
            marker = Marker()
            marker.header.frame_id = self.MAP_FRAME
            if self.use_sim_time and self.latest_clock is not None:
                marker.header.stamp = self.latest_clock
            else:
                marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'aruco_markers'
            marker.id = int(marker_id)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            pos = detection['position_odom']
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = marker.scale.y = marker.scale.z = 0.4
            
            # Color based on best camera
            color = colors.get(detection['camera'], [1.0, 1.0, 1.0])
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            
            marker_array.markers.append(marker)
            
            # Text label with information
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = 'aruco_labels'
            text_marker.id = int(marker_id) + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = marker.pose.position.x
            text_marker.pose.position.y = marker.pose.position.y
            text_marker.pose.position.z = marker.pose.position.z + 0.5
            text_marker.pose.orientation.w = 1.0
            
            text_marker.scale.z = 0.3
            text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            # Information text
            update_count = detection.get('update_count', 1)
            text_marker.text = (f"ID:{marker_id}\nU:{update_count}")
            
            marker_array.markers.append(text_marker)
        
        self.marker_pub.publish(marker_array)

    def _ensure_aruco_detections_folder(self, folder_path):
        """
        Create aruco_detections folder and images subfolder if they don't exist.
        """
        
        # Create main folder
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                self.get_logger().info(f"Created folder: {folder_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to create folder {folder_path}: {e}")
                return
        else:
            self.get_logger().debug(f"Folder already exists: {folder_path}")
        
        # Create images subfolder
        images_folder = os.path.join(folder_path, "images")
        if not os.path.exists(images_folder):
            try:
                os.makedirs(images_folder)
                self.get_logger().info(f"Created images folder: {images_folder}")
            except Exception as e:
                self.get_logger().error(f"Failed to create images folder {images_folder}: {e}")
        else:
            self.get_logger().debug(f"Images folder already exists: {images_folder}")
        

    def _crop_aruco_image(self, image, corners, padding=50):
        """
        Crop image around ArUco marker with padding.
        
        Args:
            image: RGB image
            corners: ArUco corner coordinates
            padding: Padding around marker in pixels
            
        Returns:
            np.array: Cropped image
        """
        try:
            # Get bounding box of the marker
            x_coords = corners[:, 0]
            y_coords = corners[:, 1]
            
            min_x = int(max(0, np.min(x_coords) - padding))
            max_x = int(min(image.shape[1], np.max(x_coords) + padding))
            min_y = int(max(0, np.min(y_coords) - padding))
            max_y = int(min(image.shape[0], np.max(y_coords) + padding))
            
            # Crop the image
            cropped = image[min_y:max_y, min_x:max_x]
            
            return cropped
            
        except Exception as e:
            self.get_logger().error(f"Failed to crop ArUco image: {e}")
            return None

    def _save_aruco_image(self, marker_id, image_data, update_type):
        """
        Save cropped ArUco image to images folder.
        
        Args:
            marker_id: ArUco marker ID
            image_data: Image data from detection
            update_type: "initial" or "update"
        """
        try:
            # Find best quality image for this marker
            if not image_data:
                return
                
            best_image_data = max(image_data, key=lambda x: x['quality'])
            
            # Crop the image
            cropped_image = self._crop_aruco_image(
                best_image_data['image'], 
                best_image_data['corners'], 
                padding=50
            )
            
            if cropped_image is not None:
                # Generate filename
                timestamp = int(time.time())
                filename = f"aruco_{marker_id}.jpg"
                filepath = os.path.join(self.images_folder, filename)
                
                # Convert RGB to BGR for OpenCV
                bgr_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
                
                # Save image
                cv2.imwrite(filepath, bgr_image)

                self.get_logger().info(f"Saved ArUco {marker_id}, image: {filename} in {self.images_folder}")

        except Exception as e:
            self.get_logger().error(f"Failed to save ArUco {marker_id} image: {e}")



    def gps_to_xy(self, lat, lon):
        """
        Convert GPS (latitude, longitude) to XY (meters) in Poland CS92.
        """
        x, y = self.to_xy_transformer.transform(lon, lat)
        return x, y
    

    def xy_to_gps(self, x, y):
        """
        Convert XY (meters) in Poland CS92 back to GPS (latitude, longitude).
        """
        lon, lat = self.to_gps_transformer.transform(x, y)
        return lat, lon


        
def main(args=None):
    """
    Main function to initialize and run the ArUco localization system.
    """
    rclpy.init(args=args)
    
    try:
        node = ArucoLocalization()
        node.get_logger().info("ArUco localization system started successfully")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if 'node' in locals():
                node.destroy_node()
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()