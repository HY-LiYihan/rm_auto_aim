// Copyright 2022 Chen Jun
// Copyright 2025 Yihan Li
// Licensed under the MIT License.

#include "armor_tracker/tracker.hpp"

#include <angles/angles.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>

#include <rclcpp/logger.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// STD
#include <cfloat>
#include <memory>
#include <string>

namespace rm_auto_aim {

/**
 * @brief Construct a new Tracker object.
 * Initializes the state machine and EKF parameters.
 *
 * @param max_match_distance Max distance to consider two armors as the same target.
 * @param max_match_yaw_diff Max yaw difference to consider two armors as the same target.
 */
Tracker::Tracker(double max_match_distance, double max_match_yaw_diff)
    : tracker_state(LOST),
      tracked_id(std::string("")),
      measurement(Eigen::VectorXd::Zero(4)),
      target_state(Eigen::VectorXd::Zero(9)),
      max_match_distance_(max_match_distance),
      max_match_yaw_diff_(max_match_yaw_diff) {}

/**
 * @brief Initialize the tracker with the first received armors message.
 * Selects the armor closest to the image center as the target.
 *
 * @param armors_msg The pointer to the received armors message.
 */
void Tracker::init(const Armors::SharedPtr &armors_msg) {
  if (armors_msg->armors.empty()) {
    return;
  }

  // 1. Target Selection Strategy: Closest to Image Center.
  double min_distance = DBL_MAX;
  tracked_armor = armors_msg->armors[0];
  for (const auto &armor : armors_msg->armors) {
    if (armor.distance_to_image_center < min_distance) {
      min_distance = armor.distance_to_image_center;
      tracked_armor = armor;
    }
  }

  // 2. Initialize EKF with the selected armor.
  initEKF(tracked_armor);
  RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Init EKF!");

  // 3. Update internal state.
  tracked_id = tracked_armor.number;
  tracker_state = DETECTING;

  updateArmorsNum(tracked_armor);
}

/**
 * @brief Update the tracker with new armors message.
 * Runs the Predict -> Match -> Update cycle.
 *
 * @param armors_msg The pointer to the received armors message.
 */
void Tracker::update(const Armors::SharedPtr &armors_msg) {
  // --- Step 1: EKF Prediction ---
  // Predict the target's state at the current timestamp.
  Eigen::VectorXd ekf_prediction = ekf.predict();
  RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF predict");

  bool matched = false;
  // Use prediction as default state (in case no match is found).
  target_state = ekf_prediction;

  if (!armors_msg->armors.empty()) {
    // --- Step 2: Data Association (Matching) ---
    // Find the armor that best matches the prediction.
    
    Armor same_id_armor;
    int same_id_armors_count = 0;
    auto predicted_position = getArmorPositionFromState(ekf_prediction);
    double min_position_diff = DBL_MAX;
    double yaw_diff = DBL_MAX;

    for (const auto &armor : armors_msg->armors) {
      // Only consider armors with the same ID (e.g., "1").
      if (armor.number == tracked_id) {
        same_id_armor = armor;
        same_id_armors_count++;
        
        // Calculate Euclidean distance between prediction and observation.
        auto p = armor.pose.position;
        Eigen::Vector3d position_vec(p.x, p.y, p.z);
        double position_diff = (predicted_position - position_vec).norm();
        
        if (position_diff < min_position_diff) {
          // Update the best candidate.
          min_position_diff = position_diff;
          // Calculate yaw difference. Note: state(6) is Yaw.
          yaw_diff = abs(orientationToYaw(armor.pose.orientation) - ekf_prediction(6));
          tracked_armor = armor;
        }
      }
    }

    // Store differences for debugging/tuning.
    info_position_diff = min_position_diff;
    info_yaw_diff = yaw_diff;

    // --- Step 3: Check Matching Thresholds ---
    if (min_position_diff < max_match_distance_ && yaw_diff < max_match_yaw_diff_) {
      // Case A: Matched successfully.
      matched = true;
      auto p = tracked_armor.pose.position;
      
      // Update EKF with the matched measurement.
      double measured_yaw = orientationToYaw(tracked_armor.pose.orientation);
      measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
      target_state = ekf.update(measurement);
      RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update");

    } else if (same_id_armors_count == 1 && yaw_diff > max_match_yaw_diff_) {
      // Case B: Armor Jump (Spinning Target).
      // If we see an armor with the same ID, but the yaw angle has changed significantly,
      // it means the robot has spun, and we are now seeing a different armor plate.
      handleArmorJump(same_id_armor);
    } else {
      // Case C: No match found.
      RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "No matched armor found!");
    }
  }

  // --- Step 4: Radius Constraint ---
  // The radius of the robot (from center to armor) should be within physical limits.
  // Standard infantry radius is around 20-30cm.
  if (target_state(8) < 0.12) {
    target_state(8) = 0.12;
    ekf.setState(target_state);
  } else if (target_state(8) > 0.4) {
    target_state(8) = 0.4;
    ekf.setState(target_state);
  }

  // --- Step 5: State Machine Transition ---
  if (tracker_state == DETECTING) {
    if (matched) {
      detect_count_++;
      // If we consistently match for `tracking_thres` frames, switch to TRACKING.
      if (detect_count_ > tracking_thres) {
        detect_count_ = 0;
        tracker_state = TRACKING;
      }
    } else {
      detect_count_ = 0;
      tracker_state = LOST;
    }
  } else if (tracker_state == TRACKING) {
    if (!matched) {
      tracker_state = TEMP_LOST;
      lost_count_++;
    }
  } else if (tracker_state == TEMP_LOST) {
    if (!matched) {
      lost_count_++;
      // If lost for too long (`lost_thres`), switch to LOST.
      if (lost_count_ > lost_thres) {
        lost_count_ = 0;
        tracker_state = LOST;
      }
    } else {
      tracker_state = TRACKING;
      lost_count_ = 0;
    }
  }
}

/**
 * @brief Initialize the EKF state vector based on the first observation.
 * * @param a The armor object used for initialization.
 */
void Tracker::initEKF(const Armor &a) {
  double xa = a.pose.position.x;
  double ya = a.pose.position.y;
  double za = a.pose.position.z;
  last_yaw_ = 0;
  double yaw = orientationToYaw(a.pose.orientation);

  // Initialize the state vector (9 dimensions).
  // [xc, v_xc, yc, v_yc, za, v_za, yaw, v_yaw, r]
  target_state = Eigen::VectorXd::Zero(9);
  
  // Initial Guess:
  // Assume the robot center is r=0.26m behind the armor.
  double r = 0.26;
  double xc = xa + r * cos(yaw);
  double yc = ya + r * sin(yaw);
  
  dz = 0;
  another_r = r;
  
  target_state << xc, 0, yc, 0, za, 0, yaw, 0, r;

  ekf.setState(target_state);
}

/**
 * @brief Determine the number of armors (Balance, Outpost, or Normal) based on ID and type.
 */
void Tracker::updateArmorsNum(const Armor &armor) {
  if (armor.type == "large" && (tracked_id == "3" || tracked_id == "4" || tracked_id == "5")) {
    tracked_armors_num = ArmorsNum::BALANCE_2; // 2-armor structure (Balance Infantry)
  } else if (tracked_id == "outpost") {
    tracked_armors_num = ArmorsNum::OUTPOST_3; // 3-armor structure (Outpost)
  } else {
    tracked_armors_num = ArmorsNum::NORMAL_4;  // 4-armor structure (Standard)
  }
}

/**
 * @brief Handle the situation where the tracked armor switches due to rotation.
 * This is crucial for tracking spinning targets (Top-Spin).
 */
void Tracker::handleArmorJump(const Armor &current_armor) {
  double yaw = orientationToYaw(current_armor.pose.orientation);
  target_state(6) = yaw; // Reset yaw to the new armor's yaw
  updateArmorsNum(current_armor);
  
  // Swap radius and height for 4-armor robots (which might have uneven armor placement).
  if (tracked_armors_num == ArmorsNum::NORMAL_4) {
    dz = target_state(4) - current_armor.pose.position.z;
    target_state(4) = current_armor.pose.position.z;
    std::swap(target_state(8), another_r);
  }
  
  RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "Armor jump!");

  // Safety Check:
  // If the inferred position is too far from the actual position, the EKF likely diverged.
  // In this case, perform a hard reset of the state.
  auto p = current_armor.pose.position;
  Eigen::Vector3d current_p(p.x, p.y, p.z);
  Eigen::Vector3d infer_p = getArmorPositionFromState(target_state);
  
  if ((current_p - infer_p).norm() > max_match_distance_) {
    double r = target_state(8);
    target_state(0) = p.x + r * cos(yaw);  // xc
    target_state(1) = 0;                   // vxc
    target_state(2) = p.y + r * sin(yaw);  // yc
    target_state(3) = 0;                   // vyc
    target_state(4) = p.z;                 // za
    target_state(5) = 0;                   // vza
    RCLCPP_ERROR(rclcpp::get_logger("armor_tracker"), "Reset State!");
  }

  ekf.setState(target_state);
}

/**
 * @brief Convert ROS Quaternion to Yaw angle (continuous).
 */
double Tracker::orientationToYaw(const geometry_msgs::msg::Quaternion &q) {
  tf2::Quaternion tf_q;
  tf2::fromMsg(q, tf_q);
  double roll, pitch, yaw;
  tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
  
  // Handle angle wrapping (e.g., jumping from 179 to -179).
  // angles::shortest_angular_distance ensures continuity.
  yaw = last_yaw_ + angles::shortest_angular_distance(last_yaw_, yaw);
  last_yaw_ = yaw;
  return yaw;
}

/**
 * @brief Calculate the expected armor position from the state vector.
 * Used for data association (matching).
 */
Eigen::Vector3d Tracker::getArmorPositionFromState(const Eigen::VectorXd &x) {
  // State: [xc, vxc, yc, vyc, za, vza, yaw, vyaw, r]
  double xc = x(0), yc = x(2), za = x(4);
  double yaw = x(6), r = x(8);
  
  // Calculate Armor Position (xa, ya) from Center Position (xc, yc)
  // Geometry: Armor is at distance 'r' from center at angle 'yaw + PI' (facing out)
  // Wait, let's check the coordinate definition.
  // Usually, armor faces X-axis. So if yaw=0, armor is at (xc+r, yc).
  // But here code is: xa = xc - r * cos(yaw). This implies armor is "behind" the center?
  // Let's assume standard model: 
  // Armor normal vector points out. Camera looks at armor.
  // So armor is at (xc - r*cos(yaw), yc - r*sin(yaw)).
  double xa = xc - r * cos(yaw);
  double ya = yc - r * sin(yaw);
  
  return Eigen::Vector3d(xa, ya, za);
}

}  // namespace rm_auto_aim