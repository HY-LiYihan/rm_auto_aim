// Copyright 2022 Chen Jun
// Copyright 2025 Yihan Li
// Licensed under the MIT License.

#include "armor_detector/detector.hpp"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

// STD
#include <algorithm>
#include <cmath>
#include <vector>

#include "auto_aim_interfaces/msg/debug_armor.hpp"
#include "auto_aim_interfaces/msg/debug_light.hpp"

namespace rm_auto_aim {

Detector::Detector(const int &bin_thres, const int &color, const LightParams &l,
                   const ArmorParams &a)
    : binary_thres(bin_thres), detect_color(color), l(l), a(a) {}

/**
 * @brief The main entry point for detection.
 * Pipeline: Preprocess -> Find Lights -> Match Lights -> Extract Numbers -> Classify
 */
std::vector<Armor> Detector::detect(const cv::Mat &input) {
  binary_img = preprocessImage(input);
  lights_ = findLights(input, binary_img);
  armors_ = matchLights(lights_);

  if (!armors_.empty()) {
    // Crop the number area and feed it to the neural network
    classifier->extractNumbers(input, armors_);
    classifier->classify(armors_);
  }

  return armors_;
}

/**
 * @brief Preprocesses the image using HSV V-channel extraction.
 * Modified to use the V (Value/Brightness) channel from HSV space instead of standard grayscale.
 * This ensures that Blue lights are as bright as Red lights, improving robustness.
 */
cv::Mat Detector::preprocessImage(const cv::Mat &rgb_img) {
  cv::Mat hsv_img;
  // Convert RGB to HSV color space
  cv::cvtColor(rgb_img, hsv_img, cv::COLOR_RGB2HSV);

  std::vector<cv::Mat> channels;
  // Split the image into H, S, and V channels
  cv::split(hsv_img, channels);

  cv::Mat binary_img;
  // Use the 3rd channel (Index 2), which is V (Value/Brightness)
  // V = max(R, G, B), so it works equally well for both Red and Blue armor.
  cv::threshold(channels[2], binary_img, binary_thres, 255, cv::THRESH_BINARY);

  return binary_img;
}

/**
 * @brief Finds potential light bars from the binary image.
 * OPTIMIZED: Uses cv::mean with a mask instead of pixel-wise loops.
 */
std::vector<Light> Detector::findLights(const cv::Mat &rbg_img,
                                        const cv::Mat &binary_img) {
  using std::vector;
  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;

  // RETR_EXTERNAL: Only retrieve the outer contours (ignore holes inside lights).
  cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  vector<Light> lights;
  this->debug_lights.data.clear();

  for (const auto &contour : contours) {
    // Ignore noise (too small contours)
    if (contour.size() < 5) continue;

    auto r_rect = cv::minAreaRect(contour);
    auto light = Light(r_rect);

    if (isLight(light)) {
      auto rect = light.boundingRect();

      // Safety check: ensure the bounding box is within image boundaries
      if (0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= rbg_img.cols &&
          0 <= rect.y && 0 <= rect.height && rect.y + rect.height <= rbg_img.rows) {
        
        // --- OPTIMIZATION START: Use Mask + Mean instead of Loop ---
        
        // 1. Get the Region of Interest (ROI) from the RGB image
        cv::Mat roi = rbg_img(rect);
        
        // 2. Create a black mask of the same size as ROI
        cv::Mat mask = cv::Mat::zeros(rect.size(), CV_8UC1);
        
        // 3. Transform contour points to ROI coordinate system
        // The original contour points are in the global image coordinates.
        // We need to shift them by subtracting the top-left corner of the rect.
        vector<vector<cv::Point>> roi_contours;
        vector<cv::Point> roi_poly;
        roi_poly.reserve(contour.size());
        
        cv::Point tl = rect.tl(); // Top-Left point of the bounding rect
        for (const auto &p : contour) {
          roi_poly.emplace_back(p - tl);
        }
        roi_contours.emplace_back(roi_poly);

        // 4. Fill the contour area with White (255) on the mask
        // thickness = -1 means fill the interior
        cv::drawContours(mask, roi_contours, -1, 255, -1);

        // 5. Calculate average color inside the mask area
        cv::Scalar average_color = cv::mean(roi, mask);

        // 6. Determine color based on channel intensity
        // Assuming Input is RGB: Index 0 is Red, Index 2 is Blue.
        // NOTE: Check your camera driver! If it's BGR, swap the indices.
        light.color = average_color[0] > average_color[2] ? RED : BLUE;
        
        // --- OPTIMIZATION END ---

        lights.emplace_back(light);
      }
    }
  }

  return lights;
}

/**
 * @brief Filters light bars based on geometric properties.
 */
bool Detector::isLight(const Light &light) {
  // Ratio: width / length (short side / long side)
  float ratio = light.width / light.length;
  bool ratio_ok = l.min_ratio < ratio && ratio < l.max_ratio;

  bool angle_ok = light.tilt_angle < l.max_angle;

  bool is_light = ratio_ok && angle_ok;

  // Fill in debug information for ROS topics
  auto_aim_interfaces::msg::DebugLight light_data;
  light_data.center_x = light.center.x;
  light_data.ratio = ratio;
  light_data.angle = light.tilt_angle;
  light_data.is_light = is_light;
  this->debug_lights.data.emplace_back(light_data);

  return is_light;
}

/**
 * @brief Matches pairs of lights to form armors.
 * Iterates through all possible pairs and checks consistency.
 */
std::vector<Armor> Detector::matchLights(const std::vector<Light> &lights) {
  std::vector<Armor> armors;
  this->debug_armors.data.clear();

  // Double loop to generate all pairs (O(N^2))
  for (auto light_1 = lights.begin(); light_1 != lights.end(); light_1++) {
    for (auto light_2 = light_1 + 1; light_2 != lights.end(); light_2++) {
      // 1. Color check: must match the target enemy color
      if (light_1->color != detect_color || light_2->color != detect_color)
        continue;

      // 2. Inclusion check: prevent matching a light with a reflection inside it
      if (containLight(*light_1, *light_2, lights)) {
        continue;
      }

      // 3. Geometric check: determine if it's a valid armor (Small or Large)
      auto type = isArmor(*light_1, *light_2);
      if (type != ArmorType::INVALID) {
        auto armor = Armor(*light_1, *light_2);
        armor.type = type;
        armors.emplace_back(armor);
      }
    }
  }

  return armors;
}

/**
 * @brief Checks if there are other lights inside the bounding box of a light pair.
 * Used to filter out "nested" lights (e.g., reflections on the armor plate).
 */
bool Detector::containLight(const Light &light_1, const Light &light_2,
                            const std::vector<Light> &lights) {
  auto points = std::vector<cv::Point2f>{light_1.top, light_1.bottom,
                                         light_2.top, light_2.bottom};
  auto bounding_rect = cv::boundingRect(points);

  for (const auto &test_light : lights) {
    if (test_light.center == light_1.center ||
        test_light.center == light_2.center)
      continue;

    if (bounding_rect.contains(test_light.top) ||
        bounding_rect.contains(test_light.bottom) ||
        bounding_rect.contains(test_light.center)) {
      return true;
    }
  }

  return false;
}

/**
 * @brief Validates the geometry of a light pair.
 * Checks: Length Ratio, Center Distance, Parallel Angle.
 */
ArmorType Detector::isArmor(const Light &light_1, const Light &light_2) {
  // 1. Length Ratio Check
  float light_length_ratio =
      light_1.length < light_2.length
          ? light_1.length / light_2.length
          : light_2.length / light_1.length;
  bool light_ratio_ok = light_length_ratio > a.min_light_ratio;

  // 2. Distance Check (normalized by light length)
  float avg_light_length = (light_1.length + light_2.length) / 2;
  float center_distance =
      cv::norm(light_1.center - light_2.center) / avg_light_length;
  
  bool center_distance_ok =
      (a.min_small_center_distance <= center_distance &&
       center_distance < a.max_small_center_distance) ||
      (a.min_large_center_distance <= center_distance &&
       center_distance < a.max_large_center_distance);

  // 3. Parallel Angle Check
  cv::Point2f diff = light_1.center - light_2.center;
  float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
  bool angle_ok = angle < a.max_angle;

  bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

  // Determine Type (Small vs Large) based on distance
  ArmorType type;
  if (is_armor) {
    type = center_distance > a.min_large_center_distance ? ArmorType::LARGE
                                                         : ArmorType::SMALL;
  } else {
    type = ArmorType::INVALID;
  }

  // Debug info
  auto_aim_interfaces::msg::DebugArmor armor_data;
  armor_data.type = ARMOR_TYPE_STR[static_cast<int>(type)];
  armor_data.center_x = (light_1.center.x + light_2.center.x) / 2;
  armor_data.light_ratio = light_length_ratio;
  armor_data.center_distance = center_distance;
  armor_data.angle = angle;
  this->debug_armors.data.emplace_back(armor_data);

  return type;
}

cv::Mat Detector::getAllNumbersImage() {
  if (armors_.empty()) {
    return cv::Mat(cv::Size(20, 28), CV_8UC1);
  } else {
    std::vector<cv::Mat> number_imgs;
    number_imgs.reserve(armors_.size());
    for (auto &armor : armors_) {
      number_imgs.emplace_back(armor.number_img);
    }
    cv::Mat all_num_img;
    cv::vconcat(number_imgs, all_num_img);
    return all_num_img;
  }
}

void Detector::drawResults(cv::Mat &img) {
  // Draw Lights
  for (const auto &light : lights_) {
    cv::circle(img, light.top, 3, cv::Scalar(255, 255, 255), 1);
    cv::circle(img, light.bottom, 3, cv::Scalar(255, 255, 255), 1);
    auto line_color = light.color == RED ? cv::Scalar(255, 255, 0)
                                         : cv::Scalar(255, 0, 255);
    cv::line(img, light.top, light.bottom, line_color, 1);
  }

  // Draw Armors
  for (const auto &armor : armors_) {
    cv::line(img, armor.left_light.top, armor.right_light.bottom,
             cv::Scalar(0, 255, 0), 2);
    cv::line(img, armor.left_light.bottom, armor.right_light.top,
             cv::Scalar(0, 255, 0), 2);
  }

  // Draw Confidence Text
  for (const auto &armor : armors_) {
    cv::putText(img, armor.classfication_result, armor.left_light.top,
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
  }
}

}  // namespace rm_auto_aim