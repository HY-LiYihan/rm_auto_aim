// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__ARMOR_HPP_
#define ARMOR_DETECTOR__ARMOR_HPP_

#include <opencv2/core.hpp>

// STL
#include <algorithm>
#include <string>

namespace rm_auto_aim {

// Constants for armor colors.
const int RED = 0;
const int BLUE = 1;

// Enumeration for armor types.
// SMALL: Infantry/Hero side armor.
// LARGE: Sentry/Hero front armor.
enum class ArmorType { SMALL, LARGE, INVALID };

// String representation for debugging or logging.
const std::string ARMOR_TYPE_STR[3] = {"small", "large", "invalid"};

/**
 * @brief Represents a single light bar on the armor plate.
 * Inherits from cv::RotatedRect to utilize OpenCV's geometric properties.
 */
struct Light : public cv::RotatedRect {
  Light() = default;

  /**
   * @brief Construct a new Light object from a RotatedRect.
   * Calculates the top and bottom center points and the tilt angle.
   *
   * @param box The bounding box of the light contour.
   */
  explicit Light(cv::RotatedRect box) : cv::RotatedRect(box) {
    cv::Point2f p[4];
    box.points(p);

    // Sort the 4 vertices based on their Y-coordinates (ascending).
    // The first two points will be the "top" points, the last two are "bottom".
    std::sort(p, p + 4, [](const cv::Point2f &a, const cv::Point2f &b) {
      return a.y < b.y;
    });

    // Calculate the midpoint of the top edge and bottom edge.
    top = (p[0] + p[1]) / 2;
    bottom = (p[2] + p[3]) / 2;

    // Calculate physical properties of the light bar.
    length = cv::norm(top - bottom);
    width = cv::norm(p[0] - p[1]);

    // Calculate the tilt angle relative to the vertical axis.
    // atan2(dx, dy) gives the angle from the Y-axis.
    tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
    tilt_angle = tilt_angle / CV_PI * 180;
  }

  // Color of the light (RED or BLUE).
  int color;
  
  // Center points of the top and bottom edges of the light bar.
  cv::Point2f top, bottom;
  
  // Dimensions of the light bar.
  double length;
  double width;
  
  // Tilt angle in degrees (deviation from vertical).
  float tilt_angle;
};

/**
 * @brief Represents an armor plate, consisting of two Light bars.
 */
struct Armor {
  Armor() = default;

  /**
   * @brief Construct a new Armor object from two Light bars.
   * Automatically assigns left and right lights based on X-coordinates.
   *
   * @param l1 First light bar.
   * @param l2 Second light bar.
   */
  Armor(const Light &l1, const Light &l2) {
    if (l1.center.x < l2.center.x) {
      left_light = l1;
      right_light = l2;
    } else {
      left_light = l2;
      right_light = l1;
    }
    // The armor center is the midpoint of the two lights.
    center = (left_light.center + right_light.center) / 2;
  }

  // --- Geometric Information ---
  Light left_light, right_light;
  cv::Point2f center;
  ArmorType type;

  // --- Deep Learning / Classification Info ---
  cv::Mat number_img;            // The warped image ROI for number classification.
  std::string number;            // Result (e.g., "1", "2", "G").
  float confidence;              // Classification confidence score.
  std::string classfication_result; // Full result string for debugging.
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__ARMOR_HPP_