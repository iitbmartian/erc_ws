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

#ifndef NAV_EXPLORE_H_
#define NAV_EXPLORE_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "explore/costmap_client.h"
#include "explore/frontier_search.h"

using namespace std::placeholders;

#ifdef ELOQUENT
#define ACTION_NAME "NavigateToPose"
#elif DASHING
#define ACTION_NAME "NavigateToPose"
#else
#define ACTION_NAME "navigate_to_pose"
#endif

namespace explore
{

/**
 * @class Explore
 * @brief A class adhering to the robot_actions::Action interface that moves the
 * robot base to explore its environment.
 */
class Explore : public rclcpp::Node
{
public:
  Explore();
  ~Explore();
  void start();
  void stop(bool finished_exploring = false);
  void resume();

  using NavigationGoalHandle =
    rclcpp_action::ClientGoalHandle<nav2_msgs::action::NavigateToPose>;

private:
  /**
   * @brief Make a global plan
   */
  void makePlan();

  /**
   * @brief Publish frontiers as markers
   */
  void visualizeFrontiers(
    const std::vector<frontier_exploration::Frontier>& frontiers);

  /**
   * @brief Check if goal is on blacklist
   */
  bool goalOnBlacklist(const geometry_msgs::msg::Point& goal);

  /**
   * @brief Check if goal is in obstacle
   */
  bool goalInObstacle(const geometry_msgs::msg::Point& goal);

  /**
   * @brief Check if goal is too close to boundary markers
   */
  bool goalTooCloseToMarkers(const geometry_msgs::msg::Point& goal);

  /**
   * @brief Check if robot is too close to boundary markers
   */
  bool robotTooCloseToMarkers();

  /**
   * @brief Check if goal maintains directional consistency
   */
  bool goalMaintainsDirection(const geometry_msgs::msg::Point& goal);

  /**
   * @brief Check if goal requires backtracking through explored areas
   */
  bool goalRequiresBacktracking(const geometry_msgs::msg::Point& goal);

  /**
   * @brief Handle boundary violation by robot - send escape goal
   */
  void handleBoundaryViolation();

  /**
   * @brief Find a safe escape position away from boundary markers
   */
  geometry_msgs::msg::Point findEscapePosition();

  /**
   * @brief Recovery mechanism when exploration gets stuck
   */
  void performRecovery();

  /**
   * @brief Update exploration progress tracking
   */
  void updateExplorationProgress(const geometry_msgs::msg::Point& goal);

  /**
   * @brief Callback for boundary markers
   */
  void boundaryCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg);

  /**
   * @brief Callback for robot pose
   */
  void robotPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);

  NavigationGoalHandle::SharedPtr navigation_goal_handle_;

  void reachedGoal(const NavigationGoalHandle::WrappedResult& result,
                   const geometry_msgs::msg::Point& frontier_goal);

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    marker_array_publisher_;

  rclcpp::Logger logger_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  Costmap2DClient costmap_client_;
  rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SharedPtr
    move_base_client_;
  frontier_exploration::FrontierSearch search_;
  rclcpp::TimerBase::SharedPtr exploring_timer_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr resume_subscription_;
  
  // New subscriptions for boundary awareness
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr boundary_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr robot_pose_subscription_;

  void resumeCallback(const std_msgs::msg::Bool::SharedPtr msg);

  std::vector<geometry_msgs::msg::Point> frontier_blacklist_;
  geometry_msgs::msg::Point prev_goal_;
  double prev_distance_;
  rclcpp::Time last_progress_;
  size_t last_markers_count_;
  geometry_msgs::msg::Pose initial_pose_;

  void returnToInitialPose(void);

  // Parameters
  double planner_frequency_;
  double potential_scale_, orientation_scale_, gain_scale_;
  double progress_timeout_;
  bool visualize_;
  bool return_to_init_;
  std::string robot_base_frame_;
  bool resuming_ = false;

  // New parameters for boundary awareness and directional exploration
  double boundary_distance_;
  double direction_bias_weight_;
  double backtrack_penalty_weight_;
  double min_progress_distance_;
  std::map<int, geometry_msgs::msg::Point> boundary_markers_;
  geometry_msgs::msg::Point current_robot_pose_;
  geometry_msgs::msg::Point last_valid_direction_;
  geometry_msgs::msg::Point exploration_start_pose_;
  std::vector<geometry_msgs::msg::Point> exploration_path_;
  bool has_direction_ = false;
  bool boundary_violation_ = false;
  bool escaping_boundary_ = false;  // Flag to indicate we're in escape mode
  double total_exploration_distance_;
  
  // Recovery and state management
  rclcpp::Time escape_start_time_;
  double escape_timeout_;  // Timeout for escape mode
  rclcpp::Time last_goal_time_;
  double recovery_timeout_;  // Timeout before performing recovery
  int failed_goals_count_;  // Count of consecutive failed goals
  bool exploration_finished_;
  rclcpp::Time last_planning_attempt_;
  double planning_retry_delay_;  // Delay between planning attempts after failures
};

}  // namespace explore

#endif
