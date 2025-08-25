/*********************************************************************
 * Software License Agreement (BSD License)
 * Copyright (c) 2008, Robert Bosch LLC.
 * Copyright (c) 2015-2016, Jiri Horner.
 * Copyright (c) 2021, Carlos Alvarez, Juan Galvis.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *  * Neither the name of the Jiri Horner nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#include "explore/explore.h"
#include <algorithm>
#include <functional>
#include <cmath>

inline static bool same_point(const geometry_msgs::msg::Point& one,
                             const geometry_msgs::msg::Point& two)
{
  double dx = one.x - two.x;
  double dy = one.y - two.y;
  double dist = sqrt(dx * dx + dy * dy);
  return dist < 0.01;
}

inline static double distance(const geometry_msgs::msg::Point& p1,
                             const geometry_msgs::msg::Point& p2)
{
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;
  return sqrt(dx * dx + dy * dy);
}

namespace explore
{

Explore::Explore()
  : Node("explore_node")
  , logger_(this->get_logger())
  , tf_buffer_(this->get_clock())
  , tf_listener_(tf_buffer_)
  , costmap_client_(*this, &tf_buffer_)
  , prev_distance_(0)
  , last_markers_count_(0)
  , boundary_violation_(false)
  , escaping_boundary_(false)
  , total_exploration_distance_(0.0)
  , escape_timeout_(10.0)  // 10 second timeout for escape mode
  , recovery_timeout_(5.0)  // 5 second timeout before recovery
  , failed_goals_count_(0)
  , exploration_finished_(false)
  , planning_retry_delay_(2.0)  // 2 second delay between retries
{
  double timeout;
  double min_frontier_size;

  // Declare parameters
  this->declare_parameter("planner_frequency", 1.0);
  this->declare_parameter("progress_timeout", 30.0);
  this->declare_parameter("visualize", false);
  this->declare_parameter("potential_scale", 1e-3);
  this->declare_parameter("orientation_scale", 0.0);
  this->declare_parameter("gain_scale", 1.0);
  this->declare_parameter("min_frontier_size", 0.5);
  this->declare_parameter("return_to_init", false);
  this->declare_parameter("boundary_distance", 1.0);
  this->declare_parameter("direction_bias_weight", 3.0);
  this->declare_parameter("backtrack_penalty_weight", 5.0);
  this->declare_parameter("min_progress_distance", 2.0);

  // Get parameters
  this->get_parameter("planner_frequency", planner_frequency_);
  this->get_parameter("progress_timeout", timeout);
  this->get_parameter("visualize", visualize_);
  this->get_parameter("potential_scale", potential_scale_);
  this->get_parameter("orientation_scale", orientation_scale_);
  this->get_parameter("gain_scale", gain_scale_);
  this->get_parameter("min_frontier_size", min_frontier_size);
  this->get_parameter("return_to_init", return_to_init_);
  this->get_parameter("robot_base_frame", robot_base_frame_);
  this->get_parameter("boundary_distance", boundary_distance_);
  this->get_parameter("direction_bias_weight", direction_bias_weight_);
  this->get_parameter("backtrack_penalty_weight", backtrack_penalty_weight_);
  this->get_parameter("min_progress_distance", min_progress_distance_);

  progress_timeout_ = timeout;

  // Initialize action client
  move_base_client_ = rclcpp_action::create_client<nav2_msgs::action::NavigateToPose>(
    this, ACTION_NAME);

  // Initialize frontier search
  search_ = frontier_exploration::FrontierSearch(costmap_client_.getCostmap(),
                                               potential_scale_, gain_scale_,
                                               min_frontier_size, logger_);

  // Create publishers
  if (visualize_) {
    marker_array_publisher_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>("explore/frontiers", 10);
  }

  // Create subscribers
  resume_subscription_ = this->create_subscription<std_msgs::msg::Bool>(
    "explore/resume", 10,
    std::bind(&Explore::resumeCallback, this, std::placeholders::_1));

  // Subscribe to boundary markers
  boundary_subscription_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
    "/aruco_markers", 10,
    std::bind(&Explore::boundaryCallback, this, std::placeholders::_1));

  // Subscribe to robot pose
  robot_pose_subscription_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "rtabmap/localization_pose", 10,
    std::bind(&Explore::robotPoseCallback, this, std::placeholders::_1));

  RCLCPP_INFO(logger_, "Waiting to connect to move_base nav2 server");
  move_base_client_->wait_for_action_server();
  RCLCPP_INFO(logger_, "Connected to move_base nav2 server");

  if (return_to_init_) {
    RCLCPP_INFO(logger_, "Getting initial pose of the robot");
    geometry_msgs::msg::TransformStamped transformStamped;
    std::string map_frame = costmap_client_.getGlobalFrameID();
    try {
      transformStamped = tf_buffer_.lookupTransform(
        map_frame, robot_base_frame_, tf2::TimePointZero);
      initial_pose_.position.x = transformStamped.transform.translation.x;
      initial_pose_.position.y = transformStamped.transform.translation.y;
      initial_pose_.orientation = transformStamped.transform.rotation;
      
      // Set exploration start pose
      exploration_start_pose_ = initial_pose_.position;
      current_robot_pose_ = initial_pose_.position;
      
    } catch (tf2::TransformException& ex) {
      RCLCPP_ERROR(logger_, "Couldn't find transform from %s to %s: %s",
                   map_frame.c_str(), robot_base_frame_.c_str(), ex.what());
      return_to_init_ = false;
    }
  }

  // Initialize timing
  last_goal_time_ = this->now();
  last_planning_attempt_ = this->now();

  // Start exploration timer
  exploring_timer_ = this->create_wall_timer(
    std::chrono::milliseconds((uint16_t)(1000.0 / planner_frequency_)),
    [this]() { makePlan(); });

  RCLCPP_INFO(logger_, "Exploration node initialized");
  makePlan();
}

Explore::~Explore()
{
  stop();
}

void Explore::boundaryCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
{
  RCLCPP_INFO(logger_, "Received %zu boundary markers", msg->markers.size());
  
  boundary_markers_.clear();
  for (const auto& marker : msg->markers) {
    geometry_msgs::msg::Point point;
    point.x = marker.pose.position.x;
    point.y = marker.pose.position.y;
    point.z = 0.0;
    boundary_markers_[marker.id] = point;
  }
}

void Explore::robotPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
  geometry_msgs::msg::Point new_pose = msg->pose.pose.position;
  
  // Update total exploration distance
  if (current_robot_pose_.x != 0.0 || current_robot_pose_.y != 0.0) {
    total_exploration_distance_ += distance(current_robot_pose_, new_pose);
  }
  
  current_robot_pose_ = new_pose;
  
  // Add to exploration path for backtracking prevention
  if (exploration_path_.empty() || distance(exploration_path_.back(), current_robot_pose_) > 0.5) {
    exploration_path_.push_back(current_robot_pose_);
    
    // Keep path size reasonable
    if (exploration_path_.size() > 100) {
      exploration_path_.erase(exploration_path_.begin(), exploration_path_.begin() + 20);
    }
  }
  
  // Check if robot is too close to boundary markers
  bool too_close = robotTooCloseToMarkers();
  if (too_close && !boundary_violation_) {
    boundary_violation_ = true;
    RCLCPP_ERROR(logger_, "BOUNDARY VIOLATION DETECTED! Robot too close to boundary markers!");
    handleBoundaryViolation();
  } else if (!too_close && boundary_violation_) {
    boundary_violation_ = false;
    escaping_boundary_ = false;
    RCLCPP_INFO(logger_, "Robot moved away from boundary, continuing exploration");
  }
}

bool Explore::robotTooCloseToMarkers()
{
  for (const auto& marker_pair : boundary_markers_) {
    double dist = distance(current_robot_pose_, marker_pair.second);
    if (dist < boundary_distance_) {
      RCLCPP_WARN(logger_, "Robot too close to boundary marker %d (distance: %.2f)",
                  marker_pair.first, dist);
      return true;
    }
  }
  return false;
}

geometry_msgs::msg::Point Explore::findEscapePosition()
{
  geometry_msgs::msg::Point escape_pos = current_robot_pose_;
  
  if (boundary_markers_.empty()) {
    return escape_pos;
  }
  
  // Find the closest boundary marker
  double min_dist = std::numeric_limits<double>::max();
  geometry_msgs::msg::Point closest_marker;
  
  for (const auto& marker_pair : boundary_markers_) {
    double dist = distance(current_robot_pose_, marker_pair.second);
    if (dist < min_dist) {
      min_dist = dist;
      closest_marker = marker_pair.second;
    }
  }
  
  // Calculate direction away from closest marker
  double dx = current_robot_pose_.x - closest_marker.x;
  double dy = current_robot_pose_.y - closest_marker.y;
  double magnitude = sqrt(dx * dx + dy * dy);
  
  if (magnitude > 0.01) {  // Avoid division by zero
    // Normalize direction and move to safe distance
    double safe_distance = boundary_distance_ + 0.5;  // Extra margin
    escape_pos.x = closest_marker.x + (dx / magnitude) * safe_distance;
    escape_pos.y = closest_marker.y + (dy / magnitude) * safe_distance;
  }
  
  // Check if escape position is valid (not in obstacle)
  nav2_costmap_2d::Costmap2D* costmap = costmap_client_.getCostmap();
  unsigned int mx, my;
  if (costmap->worldToMap(escape_pos.x, escape_pos.y, mx, my)) {
    unsigned char cost = costmap->getCost(mx, my);
    if (cost >= 253) {  // If escape position is in obstacle
      // Try alternative positions around the current pose
      for (int angle = 0; angle < 360; angle += 45) {
        double rad = angle * M_PI / 180.0;
        geometry_msgs::msg::Point alt_pos;
        alt_pos.x = current_robot_pose_.x + cos(rad) * 1.0;
        alt_pos.y = current_robot_pose_.y + sin(rad) * 1.0;
        
        if (costmap->worldToMap(alt_pos.x, alt_pos.y, mx, my)) {
          cost = costmap->getCost(mx, my);
          if (cost < 253) {  // Valid position
            // Check if it's far enough from all markers
            bool safe_from_all = true;
            for (const auto& marker_pair : boundary_markers_) {
              if (distance(alt_pos, marker_pair.second) < boundary_distance_) {
                safe_from_all = false;
                break;
              }
            }
            if (safe_from_all) {
              escape_pos = alt_pos;
              break;
            }
          }
        }
      }
    }
  }
  
  RCLCPP_INFO(logger_, "Calculated escape position: (%.2f, %.2f)", 
              escape_pos.x, escape_pos.y);
  return escape_pos;
}

void Explore::handleBoundaryViolation()
{
  // Cancel current goal
  if (move_base_client_) {
    move_base_client_->async_cancel_all_goals();
    RCLCPP_WARN(logger_, "Cancelled navigation goal due to boundary violation");
  }
  
  // Find and send escape goal
  geometry_msgs::msg::Point escape_position = findEscapePosition();
  
  escaping_boundary_ = true;
  escape_start_time_ = this->now();  // Record escape start time
  
  RCLCPP_WARN(logger_, "Sending escape goal to (%.2f, %.2f)", 
              escape_position.x, escape_position.y);
  
  // Send escape goal
  auto goal = nav2_msgs::action::NavigateToPose::Goal();
  goal.pose.pose.position = escape_position;
  goal.pose.pose.orientation.w = 1.;
  goal.pose.header.frame_id = costmap_client_.getGlobalFrameID();
  goal.pose.header.stamp = this->now();

  auto send_goal_options = rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SendGoalOptions();
  send_goal_options.result_callback =
    [this, escape_position](const NavigationGoalHandle::WrappedResult& result) {
      reachedGoal(result, escape_position);
    };

  move_base_client_->async_send_goal(goal, send_goal_options);
  last_goal_time_ = this->now();  // Update goal time
}

void Explore::performRecovery()
{
  RCLCPP_WARN(logger_, "Performing exploration recovery after %d failed goals", failed_goals_count_);
  
  // Clear some blacklisted goals
  if (!frontier_blacklist_.empty()) {
    size_t clear_count = std::min(size_t(5), frontier_blacklist_.size());
    frontier_blacklist_.erase(frontier_blacklist_.begin(), 
                             frontier_blacklist_.begin() + clear_count);
    RCLCPP_INFO(logger_, "Cleared %zu blacklisted goals during recovery", clear_count);
  }
  
  // Reset failed goals counter
  failed_goals_count_ = 0;
  
  // Reset escape mode if stuck
  if (escaping_boundary_) {
    RCLCPP_WARN(logger_, "Resetting escape mode during recovery");
    escaping_boundary_ = false;
    boundary_violation_ = false;
  }
  
  // Reset direction if needed to explore in different directions
  if (has_direction_ && failed_goals_count_ > 3) {
    RCLCPP_INFO(logger_, "Resetting exploration direction during recovery");
    has_direction_ = false;
  }
  
  // Force a planning attempt
  last_planning_attempt_ = this->now() - tf2::durationFromSec(planning_retry_delay_);
}

void Explore::resumeCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
  if (msg->data) {
    resume();
  } else {
    stop();
  }
}

bool Explore::goalInObstacle(const geometry_msgs::msg::Point& goal)
{
  nav2_costmap_2d::Costmap2D* costmap = costmap_client_.getCostmap();
  unsigned int mx, my;
  
  if (!costmap->worldToMap(goal.x, goal.y, mx, my)) {
    return true;  // Outside map bounds
  }
  
  unsigned char cost = costmap->getCost(mx, my);
  return cost >= 253;  // Obstacle or inscribed
}

bool Explore::goalTooCloseToMarkers(const geometry_msgs::msg::Point& goal)
{
  for (const auto& marker_pair : boundary_markers_) {
    if (distance(goal, marker_pair.second) < boundary_distance_) {
      return true;
    }
  }
  return false;
}

bool Explore::goalMaintainsDirection(const geometry_msgs::msg::Point& goal)
{
  if (!has_direction_ || total_exploration_distance_ < min_progress_distance_) {
    // First goal or haven't moved far enough yet, establish/maintain direction
    auto pose = costmap_client_.getRobotPose();
    last_valid_direction_.x = goal.x - pose.position.x;
    last_valid_direction_.y = goal.y - pose.position.y;
    has_direction_ = true;
    return true;
  }
  
  // Check if goal is in similar direction
  auto pose = costmap_client_.getRobotPose();
  geometry_msgs::msg::Point current_direction;
  current_direction.x = goal.x - pose.position.x;
  current_direction.y = goal.y - pose.position.y;
  
  // Compute dot product to check direction similarity
  double dot_product = last_valid_direction_.x * current_direction.x +
                      last_valid_direction_.y * current_direction.y;
  
  // Normalize
  double last_magnitude = sqrt(last_valid_direction_.x * last_valid_direction_.x +
                              last_valid_direction_.y * last_valid_direction_.y);
  double current_magnitude = sqrt(current_direction.x * current_direction.x +
                                 current_direction.y * current_direction.y);
  
  if (last_magnitude > 0 && current_magnitude > 0) {
    double similarity = dot_product / (last_magnitude * current_magnitude);
    
    // More strict direction consistency after traveling some distance
    double required_similarity = (total_exploration_distance_ > min_progress_distance_ * 2) ? 0.3 : 0.0;
    
    bool maintains_direction = similarity > required_similarity;
    
    if (!maintains_direction) {
      RCLCPP_DEBUG(logger_, "Goal rejected for direction inconsistency: similarity=%.2f, required=%.2f, total_dist=%.2f", 
                   similarity, required_similarity, total_exploration_distance_);
    }
    
    return maintains_direction;
  }
  
  return true;
}

bool Explore::goalRequiresBacktracking(const geometry_msgs::msg::Point& goal)
{
  if (exploration_path_.size() < 2) {
    return false;  // Not enough path history
  }
  
  // Check if goal is significantly closer to start than current position
  double goal_dist_from_start = distance(goal, exploration_start_pose_);
  double current_dist_from_start = distance(current_robot_pose_, exploration_start_pose_);
  
  // If goal is much closer to start and we've traveled a significant distance
  if (goal_dist_from_start < current_dist_from_start * 0.5 && 
      total_exploration_distance_ > min_progress_distance_) {
    RCLCPP_DEBUG(logger_, "Goal rejected for backtracking: goal_dist=%.2f, current_dist=%.2f", 
                 goal_dist_from_start, current_dist_from_start);
    return true;
  }
  
  // Check if goal would require passing near previously visited positions
  for (const auto& past_pos : exploration_path_) {
    if (distance(goal, past_pos) < 1.0) {  // Goal is near a previously visited location
      // Check if this would require significant backtracking
      double backtrack_distance = distance(current_robot_pose_, past_pos);
      if (backtrack_distance > min_progress_distance_) {
        RCLCPP_DEBUG(logger_, "Goal rejected for requiring backtracking through visited area");
        return true;
      }
    }
  }
  
  return false;
}

void Explore::updateExplorationProgress(const geometry_msgs::msg::Point& goal)
{
  // Update direction for next iteration if goal is accepted
  auto pose = costmap_client_.getRobotPose();
  last_valid_direction_.x = goal.x - pose.position.x;
  last_valid_direction_.y = goal.y - pose.position.y;
  
  RCLCPP_DEBUG(logger_, "Updated exploration direction to (%.2f, %.2f), total distance: %.2f", 
               last_valid_direction_.x, last_valid_direction_.y, total_exploration_distance_);
}

void Explore::visualizeFrontiers(
  const std::vector<frontier_exploration::Frontier>& frontiers)
{
  std_msgs::msg::ColorRGBA blue;
  blue.r = 0; blue.g = 0; blue.b = 1.0; blue.a = 1.0;
  std_msgs::msg::ColorRGBA red;
  red.r = 1.0; red.g = 0; red.b = 0; red.a = 1.0;
  std_msgs::msg::ColorRGBA green;
  green.r = 0; green.g = 1.0; green.b = 0; green.a = 1.0;
  std_msgs::msg::ColorRGBA yellow;
  yellow.r = 1.0; yellow.g = 1.0; yellow.b = 0; yellow.a = 1.0;  // For backtracking
  std_msgs::msg::ColorRGBA orange;
  orange.r = 1.0; orange.g = 0.5; orange.b = 0; orange.a = 1.0;  // For direction issues

  RCLCPP_DEBUG(logger_, "visualising %lu frontiers", frontiers.size());
  visualization_msgs::msg::MarkerArray markers_msg;
  std::vector<visualization_msgs::msg::Marker>& markers = markers_msg.markers;
  visualization_msgs::msg::Marker m;

  m.header.frame_id = costmap_client_.getGlobalFrameID();
  m.header.stamp = this->now();
  m.ns = "frontiers";
  m.scale.x = 1.0; m.scale.y = 1.0; m.scale.z = 1.0;
  m.color.r = 0; m.color.g = 0; m.color.b = 255; m.color.a = 255;

#ifdef ELOQUENT
  m.lifetime = rclcpp::Duration(0);
#elif DASHING
  m.lifetime = rclcpp::Duration(0);
#else
  m.lifetime = rclcpp::Duration::from_seconds(0);
#endif

  m.frame_locked = true;
  double min_cost = frontiers.empty() ? 0. : frontiers.front().cost;
  m.action = visualization_msgs::msg::Marker::ADD;
  size_t id = 0;

  for (auto& frontier : frontiers) {
    m.type = visualization_msgs::msg::Marker::POINTS;
    m.id = int(id);
    m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 0.1;
    m.points = frontier.points;

    if (goalOnBlacklist(frontier.centroid)) {
      m.color = red;
    } else if (goalInObstacle(frontier.centroid)) {
      m.color = red;
    } else if (goalTooCloseToMarkers(frontier.centroid)) {
      m.color = red;
    } else if (goalRequiresBacktracking(frontier.centroid)) {
      m.color = yellow;  // Yellow for backtracking frontiers
    } else if (!goalMaintainsDirection(frontier.centroid)) {
      m.color = orange;  // Orange for direction inconsistent frontiers
    } else {
      m.color = blue;
    }

    markers.push_back(m);
    ++id;

    m.type = visualization_msgs::msg::Marker::SPHERE;
    m.id = int(id);
    m.pose.position = frontier.initial;
    double scale = std::min(std::abs(min_cost * 0.4 / frontier.cost), 0.5);
    m.scale.x = scale; m.scale.y = scale; m.scale.z = scale;
    m.points = {};
    m.color = green;
    markers.push_back(m);
    ++id;
  }

  size_t current_markers_count = markers.size();
  m.action = visualization_msgs::msg::Marker::DELETE;
  for (; id < last_markers_count_; ++id) {
    m.id = int(id);
    markers.push_back(m);
  }

  last_markers_count_ = current_markers_count;
  marker_array_publisher_->publish(markers_msg);
}

void Explore::makePlan()
{
  // Check if we should delay planning after failures
  if ((this->now() - last_planning_attempt_) < tf2::durationFromSec(planning_retry_delay_)) {
    return;
  }
  
  last_planning_attempt_ = this->now();
  
  // Check for escape mode timeout
  if (escaping_boundary_ && 
      (this->now() - escape_start_time_) > tf2::durationFromSec(escape_timeout_)) {
    RCLCPP_WARN(logger_, "Escape mode timeout, forcing recovery");
    escaping_boundary_ = false;
    boundary_violation_ = false;
    performRecovery();
  }
  
  // Check for recovery timeout (no goals sent for a while)
  if ((this->now() - last_goal_time_) > tf2::durationFromSec(recovery_timeout_)) {
    RCLCPP_WARN(logger_, "No goals sent recently, performing recovery");
    performRecovery();
    last_goal_time_ = this->now();  // Reset timer
  }
  
  // If we're escaping from boundary, skip normal planning but allow escape goals
  if (escaping_boundary_) {
    RCLCPP_INFO_THROTTLE(logger_, *this->get_clock(), 2000, 
                         "Currently escaping boundary violation, skipping normal planning");
    return;
  }
  
  // Get current robot pose
  auto pose = costmap_client_.getRobotPose();
  
  // Update frontier search with boundary and direction information
  search_.setBoundaryInfo(boundary_markers_, boundary_distance_, 
                         last_valid_direction_, direction_bias_weight_, has_direction_,
                         exploration_path_, backtrack_penalty_weight_);

  // Find frontiers
  auto frontiers = search_.searchFrom(pose.position);
  RCLCPP_DEBUG(logger_, "found %lu frontiers", frontiers.size());

  for (size_t i = 0; i < frontiers.size(); ++i) {
    RCLCPP_DEBUG(logger_, "frontier %zd cost: %f", i, frontiers[i].cost);
  }

  if (frontiers.empty()) {
    RCLCPP_WARN(logger_, "No frontiers found, exploration complete!");
    exploration_finished_ = true;
    stop(true);
    return;
  }

  // Visualize frontiers
  if (visualize_) {
    visualizeFrontiers(frontiers);
  }

  // Find first valid frontier with stricter validation
  auto frontier = std::find_if(frontiers.begin(), frontiers.end(),
    [this](const frontier_exploration::Frontier& f) {
      return !goalOnBlacklist(f.centroid) && 
             !goalInObstacle(f.centroid) &&
             !goalTooCloseToMarkers(f.centroid) &&
             !goalRequiresBacktracking(f.centroid) &&
             goalMaintainsDirection(f.centroid);
    });

  if (frontier == frontiers.end()) {
    // Try without direction constraint if we're stuck
    frontier = std::find_if(frontiers.begin(), frontiers.end(),
      [this](const frontier_exploration::Frontier& f) {
        return !goalOnBlacklist(f.centroid) && 
               !goalInObstacle(f.centroid) &&
               !goalTooCloseToMarkers(f.centroid) &&
               !goalRequiresBacktracking(f.centroid);
      });
      
    if (frontier != frontiers.end()) {
      RCLCPP_WARN(logger_, "Relaxing direction constraint to find valid frontier");
    }
  }

  if (frontier == frontiers.end()) {
    RCLCPP_WARN(logger_, "No valid frontiers found. Performing recovery.");
    failed_goals_count_++;
    
    if (failed_goals_count_ >= 3) {
      performRecovery();
    } else {
      // Clear some blacklisted goals for next attempt
      if (!frontier_blacklist_.empty()) {
        size_t clear_count = std::min(size_t(2), frontier_blacklist_.size());
        frontier_blacklist_.erase(frontier_blacklist_.begin(), 
                                 frontier_blacklist_.begin() + clear_count);
        RCLCPP_INFO(logger_, "Cleared %zu blacklisted goals, will retry", clear_count);
      }
    }
    return;
  }

  geometry_msgs::msg::Point target_position = frontier->centroid;

  // Check for progress timeout
  bool same_goal = same_point(prev_goal_, target_position);
  prev_goal_ = target_position;

  if (!same_goal || prev_distance_ > frontier->min_distance) {
    last_progress_ = this->now();
    prev_distance_ = frontier->min_distance;
  }

  if ((this->now() - last_progress_ > tf2::durationFromSec(progress_timeout_)) && !resuming_) {
    frontier_blacklist_.push_back(target_position);
    RCLCPP_DEBUG(logger_, "Adding current goal to black list");
    failed_goals_count_++;
    makePlan();
    return;
  }

  if (resuming_) {
    resuming_ = false;
  }

  if (same_goal) {
    return;
  }

  // Reset failed goals counter on successful goal generation
  failed_goals_count_ = 0;

  // Update exploration progress
  updateExplorationProgress(target_position);

  RCLCPP_INFO(logger_, "Sending goal to (%.2f, %.2f), total exploration distance: %.2f", 
              target_position.x, target_position.y, total_exploration_distance_);

  // Send goal
  auto goal = nav2_msgs::action::NavigateToPose::Goal();
  goal.pose.pose.position = target_position;
  goal.pose.pose.orientation.w = 1.;
  goal.pose.header.frame_id = costmap_client_.getGlobalFrameID();
  goal.pose.header.stamp = this->now();

  auto send_goal_options = rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SendGoalOptions();
  send_goal_options.result_callback =
    [this, target_position](const NavigationGoalHandle::WrappedResult& result) {
      reachedGoal(result, target_position);
    };

  move_base_client_->async_send_goal(goal, send_goal_options);
  last_goal_time_ = this->now();  // Update goal time
}

void Explore::returnToInitialPose()
{
  if (!return_to_init_) {
    RCLCPP_INFO(logger_, "Return to initial pose disabled");
    return;
  }
  
  RCLCPP_INFO(logger_, "Exploration complete! Returning to initial pose at (%.2f, %.2f)", 
              initial_pose_.position.x, initial_pose_.position.y);
  
  auto goal = nav2_msgs::action::NavigateToPose::Goal();
  goal.pose.pose.position = initial_pose_.position;
  goal.pose.pose.orientation = initial_pose_.orientation;
  goal.pose.header.frame_id = costmap_client_.getGlobalFrameID();
  goal.pose.header.stamp = this->now();

  auto send_goal_options = rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SendGoalOptions();
  send_goal_options.result_callback =
    [this](const NavigationGoalHandle::WrappedResult& result) {
      switch (result.code) {
        case rclcpp_action::ResultCode::SUCCEEDED:
          RCLCPP_INFO(logger_, "Successfully returned to initial position!");
          break;
        case rclcpp_action::ResultCode::ABORTED:
          RCLCPP_WARN(logger_, "Failed to return to initial position - goal aborted");
          // Try again after a delay
          std::this_thread::sleep_for(std::chrono::seconds(2));
          returnToInitialPose();
          break;
        case rclcpp_action::ResultCode::CANCELED:
          RCLCPP_INFO(logger_, "Return to initial position was canceled");
          break;
        default:
          RCLCPP_WARN(logger_, "Unknown result code while returning to initial position");
          break;
      }
    };
    
  move_base_client_->async_send_goal(goal, send_goal_options);
}

bool Explore::goalOnBlacklist(const geometry_msgs::msg::Point& goal)
{
  constexpr static size_t tolerance = 5;
  nav2_costmap_2d::Costmap2D* costmap2d = costmap_client_.getCostmap();

  for (auto& frontier_goal : frontier_blacklist_) {
    double x_diff = fabs(goal.x - frontier_goal.x);
    double y_diff = fabs(goal.y - frontier_goal.y);

    if (x_diff < tolerance * costmap2d->getResolution() &&
        y_diff < tolerance * costmap2d->getResolution())
      return true;
  }
  return false;
}

void Explore::reachedGoal(const NavigationGoalHandle::WrappedResult& result,
                         const geometry_msgs::msg::Point& frontier_goal)
{
  switch (result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      if (escaping_boundary_) {
        RCLCPP_INFO(logger_, "Successfully escaped boundary violation");
        escaping_boundary_ = false;
      } else {
        RCLCPP_INFO(logger_, "Goal reached successfully");
      }
      failed_goals_count_ = 0;  // Reset failed counter on success
      break;
      
    case rclcpp_action::ResultCode::ABORTED:
      RCLCPP_DEBUG(logger_, "Goal was aborted");
      if (!escaping_boundary_) {  // Only blacklist normal exploration goals
        frontier_blacklist_.push_back(frontier_goal);
      }
      failed_goals_count_++;
      
      // If too many failures, trigger recovery
      if (failed_goals_count_ >= 3) {
        RCLCPP_WARN(logger_, "Too many goal failures, performing recovery");
        performRecovery();
      }
      break;
      
    case rclcpp_action::ResultCode::CANCELED:
      RCLCPP_DEBUG(logger_, "Goal was canceled");
      if (escaping_boundary_) {
        // If escape goal was cancelled, force recovery
        RCLCPP_WARN(logger_, "Escape goal was cancelled, forcing recovery");
        escaping_boundary_ = false;
        boundary_violation_ = false;
        performRecovery();
      }
      break;
      
    default:
      RCLCPP_WARN(logger_, "Unknown result code from move base nav2");
      failed_goals_count_++;
      break;
  }

  // Always try to continue planning unless exploration is finished
  if (!exploration_finished_) {
    // Small delay before next planning attempt to allow system to settle
    last_planning_attempt_ = this->now() + tf2::durationFromSec(1.0);
    makePlan();
  }
}

void Explore::start()
{
  RCLCPP_INFO(logger_, "Exploration started.");
  exploration_finished_ = false;
  failed_goals_count_ = 0;
  last_goal_time_ = this->now();
}

void Explore::stop(bool finished_exploring)
{
  RCLCPP_INFO(logger_, "Exploration stopped. Total distance traveled: %.2f", total_exploration_distance_);
  move_base_client_->async_cancel_all_goals();
  exploring_timer_->cancel();
  
  if (finished_exploring) {
    exploration_finished_ = true;
    if (return_to_init_) {
      // Add a small delay before returning to ensure robot has stopped
      std::this_thread::sleep_for(std::chrono::seconds(2));
      returnToInitialPose();
    }
  }
}

void Explore::resume()
{
  resuming_ = true;
  boundary_violation_ = false;  // Reset boundary violation on resume
  escaping_boundary_ = false;   // Reset escape mode on resume
  failed_goals_count_ = 0;      // Reset failure counter
  exploration_finished_ = false; // Reset finished flag
  
  RCLCPP_INFO(logger_, "Exploration resuming.");
  exploring_timer_->reset();
  last_goal_time_ = this->now();
  last_planning_attempt_ = this->now();
  makePlan();
}

}  // namespace explore

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<explore::Explore>());
  rclcpp::shutdown();
  return 0;
}
