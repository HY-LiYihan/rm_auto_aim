// Copyright 2022 Chen Jun
// Copyright 2025 Venom
// Licensed under the MIT License.

#include "armor_detector/pnp_solver.hpp"

#include <opencv2/calib3d.hpp>
#include <vector>

namespace rm_auto_aim {

/**
 * @brief Construct a new PnPSolver object.
 * Initializes the camera matrix, distortion coefficients, and the 3D model points
 * of the armor plates.
 *
 * @param camera_matrix The 3x3 intrinsic matrix of the camera (fx, fy, cx, cy).
 * @param dist_coeffs The distortion coefficients (k1, k2, p1, p2, k3).
 */
PnPSolver::PnPSolver(const std::array<double, 9> &camera_matrix,
                     const std::vector<double> &dist_coeffs)
    : camera_matrix_(cv::Mat(3, 3, CV_64F, const_cast<double *>(camera_matrix.data())).clone()),
      dist_coeffs_(cv::Mat(1, 5, CV_64F, const_cast<double *>(dist_coeffs.data())).clone()) {
  // Define the half-sizes of the armor plates in meters.
  // Converting millimeters to meters for standard ROS unit compliance.
  // Y-axis represents the width (horizontal), Z-axis represents the height (vertical).
  constexpr double small_half_y = SMALL_ARMOR_WIDTH / 2.0 / 1000.0;
  constexpr double small_half_z = SMALL_ARMOR_HEIGHT / 2.0 / 1000.0;
  constexpr double large_half_y = LARGE_ARMOR_WIDTH / 2.0 / 1000.0;
  constexpr double large_half_z = LARGE_ARMOR_HEIGHT / 2.0 / 1000.0;

  // Define 3D object points in the Armor Coordinate System.
  // The origin (0,0,0) is the center of the armor plate.
  // Coordinate definition: X-Forward, Y-Left, Z-Up.
  //
  // The order of points MUST match the order of 2D image points extraction!
  // Order: Bottom-Left -> Top-Left -> Top-Right -> Bottom-Right.

  // 1. Small Armor Points
  small_armor_points_.emplace_back(cv::Point3f(0, small_half_y, -small_half_z));  // Bottom-Left
  small_armor_points_.emplace_back(cv::Point3f(0, small_half_y, small_half_z));   // Top-Left
  small_armor_points_.emplace_back(cv::Point3f(0, -small_half_y, small_half_z));  // Top-Right
  small_armor_points_.emplace_back(cv::Point3f(0, -small_half_y, -small_half_z)); // Bottom-Right

  // 2. Large Armor Points
  large_armor_points_.emplace_back(cv::Point3f(0, large_half_y, -large_half_z));  // Bottom-Left
  large_armor_points_.emplace_back(cv::Point3f(0, large_half_y, large_half_z));   // Top-Left
  large_armor_points_.emplace_back(cv::Point3f(0, -large_half_y, large_half_z));  // Top-Right
  large_armor_points_.emplace_back(cv::Point3f(0, -large_half_y, -large_half_z)); // Bottom-Right
}

/**
 * @brief Solves the Perspective-n-Point (PnP) problem to find the armor's pose.
 *
 * @param armor The detected armor object containing 2D image points.
 * @param rvec Output: Rotation vector.
 * @param tvec Output: Translation vector (Position of armor in camera frame).
 * @return true If PnP solving was successful.
 * @return false Otherwise.
 */
bool PnPSolver::solvePnP(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec) {
  std::vector<cv::Point2f> image_armor_points;

  // Fill in 2D image points extracted from the detector.
  // The order MUST match the 3D object points defined in the constructor.
  // Order: Bottom-Left -> Top-Left -> Top-Right -> Bottom-Right.
  image_armor_points.emplace_back(armor.left_light.bottom);
  image_armor_points.emplace_back(armor.left_light.top);
  image_armor_points.emplace_back(armor.right_light.top);
  image_armor_points.emplace_back(armor.right_light.bottom);

  // Select the correct 3D model based on the armor type.
  auto object_points = armor.type == ArmorType::SMALL ? small_armor_points_
                                                      : large_armor_points_;

  // Call OpenCV's solvePnP.
  // We use SOLVEPNP_IPPE (Infinitesimal Plane-based Pose Estimation)
  // because it is robust and fast for 4 coplanar points.
  return cv::solvePnP(object_points, image_armor_points, camera_matrix_,
                      dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_IPPE);
}

/**
 * @brief Calculates the distance from the armor center to the image center.
 * Used for target selection strategy (e.g., shoot the closest one to the crosshair).
 *
 * @param image_point The 2D center point of the armor in the image.
 * @return float The distance in pixels.
 */
float PnPSolver::calculateDistanceToCenter(const cv::Point2f &image_point) {
  // Extract principal point (cx, cy) from the camera matrix.
  // K = [fx, 0, cx; 0, fy, cy; 0, 0, 1]
  float cx = camera_matrix_.at<double>(0, 2);
  float cy = camera_matrix_.at<double>(1, 2);

  // Calculate Euclidean distance.
  return cv::norm(image_point - cv::Point2f(cx, cy));
}

}  // namespace rm_auto_aim