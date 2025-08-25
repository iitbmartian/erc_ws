#include "explore/frontier_search.h"
#include "explore/costmap_tools.h"
#include <mutex>
#include <algorithm>
#include <functional>
#include <queue>
#include "nav2_costmap_2d/cost_values.hpp"

namespace frontier_exploration
{

using nav2_costmap_2d::FREE_SPACE;
using nav2_costmap_2d::LETHAL_OBSTACLE;
using nav2_costmap_2d::NO_INFORMATION;

FrontierSearch::FrontierSearch(nav2_costmap_2d::Costmap2D* costmap,
                               double potential_scale, double gain_scale,
                               double min_frontier_size, rclcpp::Logger logger)
  : costmap_(costmap)
  , potential_scale_(potential_scale)
  , gain_scale_(gain_scale)
  , min_frontier_size_(min_frontier_size)
  , logger_(logger)
  , boundary_distance_(1.0)
  , direction_weight_(2.0)
  , has_direction_(false)
  , backtrack_penalty_weight_(5.0)
{
}

void FrontierSearch::setBoundaryInfo(const std::map<int, geometry_msgs::msg::Point>& markers,
                                     double boundary_distance,
                                     const geometry_msgs::msg::Point& preferred_direction,
                                     double direction_weight,
                                     bool has_direction,
                                     const std::vector<geometry_msgs::msg::Point>& exploration_path,
                                     double backtrack_penalty_weight)
{
  boundary_markers_ = markers;
  boundary_distance_ = boundary_distance;
  preferred_direction_ = preferred_direction;
  direction_weight_ = direction_weight;
  has_direction_ = has_direction;
  exploration_path_ = exploration_path;
  backtrack_penalty_weight_ = backtrack_penalty_weight;
}

std::vector<Frontier> FrontierSearch::searchFrom(geometry_msgs::msg::Point position)
{
  std::vector<Frontier> frontier_list;

  // Sanity check that robot is inside costmap bounds before searching
  unsigned int mx, my;
  if (!costmap_->worldToMap(position.x, position.y, mx, my)) {
    RCLCPP_ERROR(logger_, "[FrontierSearch] Robot out of costmap bounds, cannot search for frontiers");
    return frontier_list;
  }

  // Make sure map is consistent and locked for duration of search
  std::lock_guard<nav2_costmap_2d::Costmap2D::mutex_t> lock(*(costmap_->getMutex()));

  map_ = costmap_->getCharMap();
  size_x_ = costmap_->getSizeInCellsX();
  size_y_ = costmap_->getSizeInCellsY();

  // Initialize flag arrays to keep track of visited and frontier cells
  std::vector<unsigned char> frontier_flag(size_x_ * size_y_, false);
  std::vector<unsigned char> visited_flag(size_x_ * size_y_, false);

  // Initialize breadth first search
  std::queue<unsigned int> bfs;

  // Find closest clear cell to start search
  unsigned int clear, pos = costmap_->getIndex(mx, my);
  if (nearestCell(clear, pos, FREE_SPACE, *costmap_)) {
    bfs.push(clear);
  } else {
    bfs.push(pos);
    RCLCPP_WARN(logger_, "[FrontierSearch] Could not find nearby clear cell to start search");
  }

  visited_flag[bfs.front()] = true;

  while (!bfs.empty()) {
    unsigned int idx = bfs.front();
    bfs.pop();

    // Iterate over 4-connected neighbourhood
    for (unsigned nbr : nhood4(idx, *costmap_)) {
      // Add to queue all free, unvisited cells, use descending search in case
      // initialized on non-free cell
      if (map_[nbr] <= map_[idx] && !visited_flag[nbr]) {
        visited_flag[nbr] = true;
        bfs.push(nbr);
      }
      // Check if cell is new frontier cell (unvisited, NO_INFORMATION, free neighbour)
      else if (isNewFrontierCell(nbr, frontier_flag)) {
        frontier_flag[nbr] = true;
        Frontier new_frontier = buildNewFrontier(nbr, pos, frontier_flag);
        if (new_frontier.size * costmap_->getResolution() >= min_frontier_size_) {
          frontier_list.push_back(new_frontier);
        }
      }
    }
  }

  // Set costs of frontiers
  for (auto& frontier : frontier_list) {
    frontier.cost = frontierCost(frontier);
  }

  std::sort(frontier_list.begin(), frontier_list.end(),
           [](const Frontier& f1, const Frontier& f2) { return f1.cost < f2.cost; });

  return frontier_list;
}

Frontier FrontierSearch::buildNewFrontier(unsigned int initial_cell,
                                         unsigned int reference,
                                         std::vector<unsigned char>& frontier_flag)
{
  // Initialize frontier structure
  Frontier output;
  output.centroid.x = 0;
  output.centroid.y = 0;
  output.size = 1;
  output.min_distance = std::numeric_limits<double>::infinity();

  // Record initial contact point for frontier
  unsigned int ix, iy;
  costmap_->indexToCells(initial_cell, ix, iy);
  costmap_->mapToWorld(ix, iy, output.initial.x, output.initial.y);

  // Push initial gridcell onto queue
  std::queue<unsigned int> bfs;
  bfs.push(initial_cell);

  // Cache reference position in world coords
  unsigned int rx, ry;
  double reference_x, reference_y;
  costmap_->indexToCells(reference, rx, ry);
  costmap_->mapToWorld(rx, ry, reference_x, reference_y);

  while (!bfs.empty()) {
    unsigned int idx = bfs.front();
    bfs.pop();

    // Try adding cells in 8-connected neighborhood to frontier
    for (unsigned int nbr : nhood8(idx, *costmap_)) {
      // Check if neighbour is a potential frontier cell
      if (isNewFrontierCell(nbr, frontier_flag)) {
        // Mark cell as frontier
        frontier_flag[nbr] = true;
        unsigned int mx, my;
        double wx, wy;
        costmap_->indexToCells(nbr, mx, my);
        costmap_->mapToWorld(mx, my, wx, wy);

        geometry_msgs::msg::Point point;
        point.x = wx;
        point.y = wy;
        output.points.push_back(point);

        // Update frontier size
        output.size++;

        // Update centroid of frontier
        output.centroid.x += wx;
        output.centroid.y += wy;

        // Determine frontier's distance from robot, going by closest gridcell to robot
        double distance = sqrt(pow((reference_x - wx), 2.0) + pow((reference_y - wy), 2.0));
        if (distance < output.min_distance) {
          output.min_distance = distance;
          output.middle.x = wx;
          output.middle.y = wy;
        }

        // Add to queue for breadth first search
        bfs.push(nbr);
      }
    }
  }

  // Average out frontier centroid
  output.centroid.x /= output.size;
  output.centroid.y /= output.size;

  return output;
}

bool FrontierSearch::isNewFrontierCell(unsigned int idx,
                                       const std::vector<unsigned char>& frontier_flag)
{
  // Check that cell is unknown and not already marked as frontier
  if (map_[idx] != NO_INFORMATION || frontier_flag[idx]) {
    return false;
  }

  // Frontier cells should have at least one cell in 4-connected neighbourhood that is free
  for (unsigned int nbr : nhood4(idx, *costmap_)) {
    if (map_[nbr] == FREE_SPACE) {
      return true;
    }
  }

  return false;
}

double FrontierSearch::frontierCost(const Frontier& frontier)
{
  // Base cost: distance penalty minus size gain
  double base_cost = (potential_scale_ * frontier.min_distance * costmap_->getResolution()) -
                     (gain_scale_ * frontier.size * costmap_->getResolution());

  // Boundary penalty: increase cost if too close to boundary markers
  double boundary_penalty = 0.0;
  for (const auto& marker_pair : boundary_markers_) {
    double dist_to_marker = sqrt(pow(frontier.centroid.x - marker_pair.second.x, 2) +
                                pow(frontier.centroid.y - marker_pair.second.y, 2));
    if (dist_to_marker < boundary_distance_) {
      boundary_penalty += (boundary_distance_ - dist_to_marker) * 15.0;  // Increased penalty
    }
  }

  // Direction consistency bonus/penalty
  double direction_cost = 0.0;
  if (has_direction_) {
    // Calculate similarity to preferred direction
    double preferred_mag = sqrt(preferred_direction_.x * preferred_direction_.x + 
                               preferred_direction_.y * preferred_direction_.y);
    if (preferred_mag > 0) {
      // Direction to this frontier from robot
      double frontier_dx = frontier.centroid.x;  // Relative to robot (at origin in cost calc)
      double frontier_dy = frontier.centroid.y;
      double frontier_mag = sqrt(frontier_dx * frontier_dx + frontier_dy * frontier_dy);
      
      if (frontier_mag > 0) {
        double dot_product = (preferred_direction_.x * frontier_dx + 
                             preferred_direction_.y * frontier_dy);
        double similarity = dot_product / (preferred_mag * frontier_mag);
        
        // Stronger penalty for opposite direction
        direction_cost = direction_weight_ * (1.5 - similarity);  // Increased penalty range
      }
    }
  }

  // Backtracking penalty: penalize frontiers near previously visited locations
  double backtrack_penalty = 0.0;
  if (!exploration_path_.empty()) {
    double min_dist_to_path = std::numeric_limits<double>::max();
    for (const auto& past_pos : exploration_path_) {
      double dist = sqrt(pow(frontier.centroid.x - past_pos.x, 2) + 
                        pow(frontier.centroid.y - past_pos.y, 2));
      min_dist_to_path = std::min(min_dist_to_path, dist);
    }
    
    // Heavy penalty if frontier is very close to previously visited areas
    if (min_dist_to_path < 2.0) {
      backtrack_penalty = backtrack_penalty_weight_ * (2.0 - min_dist_to_path);
    }
  }

  return base_cost + boundary_penalty + direction_cost + backtrack_penalty;
}

}  // namespace frontier_exploration
