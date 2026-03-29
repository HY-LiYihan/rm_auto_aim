// Copyright 2022 Chen Jun
// Copyright 2025 Yihan Li
// Licensed under the MIT License.

#include "armor_detector/detector_node.hpp"

// ROS 2 & TF2
#include <cv_bridge/cv_bridge.h>
#include <rmw/qos_profiles.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/qos.hpp>

// OpenCV
#include <opencv2/calib3d.hpp>  // Added for Rodrigues
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// STD
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"

namespace rm_auto_aim {

ArmorDetectorNode::ArmorDetectorNode(const rclcpp::NodeOptions &options)
    : Node("armor_detector", options) {
  RCLCPP_INFO(this->get_logger(), "Starting DetectorNode!");

  camera_info_topic_ = this->declare_parameter("camera_info_topic",
                                               std::string("/camera_info"));
  image_topic_ =
      this->declare_parameter("image_topic", std::string("/image_raw"));
  armors_topic_ = this->declare_parameter("armors_topic",
                                         std::string("/detector/armors"));
  marker_topic_ =
      this->declare_parameter("marker_topic", std::string("/detector/marker"));
  debug_lights_topic_ = this->declare_parameter(
      "debug_lights_topic", std::string("/detector/debug_lights"));
  debug_armors_topic_ = this->declare_parameter(
      "debug_armors_topic", std::string("/detector/debug_armors"));
  debug_binary_img_topic_ = this->declare_parameter(
      "debug_binary_img_topic", std::string("/detector/binary_img"));
  debug_number_img_topic_ = this->declare_parameter(
      "debug_number_img_topic", std::string("/detector/number_img"));
  debug_result_img_topic_ = this->declare_parameter(
      "debug_result_img_topic", std::string("/detector/result_img"));
  debug_publish_rate_ = this->declare_parameter("debug_publish_rate", 10.0);
  last_debug_pub_time_ = this->now();

  // 1. Initialize the Detector object (load params and model)
  detector_ = initDetector();

  // 2. Initialize Armors Publisher
  // Use SensorDataQoS (Best Effort) for high-frequency data to reduce latency.
  armors_pub_ = this->create_publisher<auto_aim_interfaces::msg::Armors>(
      armors_topic_, rclcpp::SensorDataQoS());

  // 3. Initialize Visualization Markers (For Rviz)
  // Armor Marker (Cube representing the armor plate)
  armor_marker_.ns = "armors";
  armor_marker_.action = visualization_msgs::msg::Marker::ADD;
  armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
  armor_marker_.scale.x = 0.03;   // Thickness
  armor_marker_.scale.z = 0.125;  // Height
  armor_marker_.color.a = 1.0;    // Alpha
  armor_marker_.color.g = 0.5;    // Greenish
  armor_marker_.color.b = 1.0;    // Blueish
  armor_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

  // Text Marker (Displays the classification result, e.g., "1", "2")
  text_marker_.ns = "classification";
  text_marker_.action = visualization_msgs::msg::Marker::ADD;
  text_marker_.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  text_marker_.scale.z = 0.1;  // Text height
  text_marker_.color.a = 1.0;
  text_marker_.color.r = 1.0;
  text_marker_.color.g = 1.0;
  text_marker_.color.b = 1.0;
  text_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      marker_topic_, 10);

  // 4. Handle Debug Mode
  debug_ = this->declare_parameter("debug", false);
  if (debug_) {
    createDebugPublishers();
  }

  // Register callback to handle dynamic parameter changes (hot reload)
  param_cb_handle_ = this->add_on_set_parameters_callback(
      std::bind(&ArmorDetectorNode::onSetParameters, this,
                std::placeholders::_1));

  // 5. Subscribe to Camera Info
  // This is a "one-shot" subscription. Once we get the intrinsics, we initialize
  // the PnP solver and shut down this subscriber to save bandwidth.
  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
        if (camera_info->k[0] <= 0.0 || camera_info->k[4] <= 0.0) {
          RCLCPP_WARN_THROTTLE(
              this->get_logger(), *this->get_clock(), 2000,
              "Invalid camera intrinsics in camera_info (fx/fy <= 0). Waiting "
              "for valid camera_info...");
          return;
        }
        if (camera_info->d.size() < 5) {
          RCLCPP_WARN_THROTTLE(
              this->get_logger(), *this->get_clock(), 2000,
              "Invalid camera distortion coefficients in camera_info "
              "(size < 5). Waiting for valid camera_info...");
          return;
        }
        if (camera_info->header.frame_id.empty()) {
          RCLCPP_WARN_THROTTLE(
              this->get_logger(), *this->get_clock(), 5000,
              "camera_info frame_id is empty. Pose frame will follow image "
              "header.frame_id.");
        }
        cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
        cam_info_ =
            std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
        // Initialize PnP Solver with camera intrinsics (K) and distortion
        // coeffs (D)
        pnp_solver_ =
            std::make_unique<PnPSolver>(camera_info->k, camera_info->d);
        cam_info_sub_.reset();  // Unsubscribe after initialization
      });

  // 6. Subscribe to Image Raw
  img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      std::bind(&ArmorDetectorNode::imageCallback, this,
                std::placeholders::_1));
}

void ArmorDetectorNode::imageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr img_msg) {
  // Detect armors from the image
  auto armors = detectArmors(img_msg);

  // If PnP solver is not ready (camera_info not received yet), skip.
  if (pnp_solver_ == nullptr) {
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "PnP solver is not initialized yet. Waiting for valid camera_info...");
    return;
  }

  // Synchronize headers
  armors_msg_.header = armor_marker_.header = text_marker_.header =
      img_msg->header;
  armors_msg_.armors.clear();
  marker_array_.markers.clear();
  armor_marker_.id = 0;
  text_marker_.id = 0;

  auto_aim_interfaces::msg::Armor armor_msg;
  for (const auto &armor : armors) {
    cv::Mat rvec, tvec;
    bool success = pnp_solver_->solvePnP(armor, rvec, tvec);

    if (success) {
      // Fill basic info
      armor_msg.type = ARMOR_TYPE_STR[static_cast<int>(armor.type)];
      armor_msg.number = armor.number;

      // Fill pose (Position)
      armor_msg.pose.position.x = tvec.at<double>(0);
      armor_msg.pose.position.y = tvec.at<double>(1);
      armor_msg.pose.position.z = tvec.at<double>(2);

      // Fill pose (Orientation): Rotation Vector -> Rotation Matrix ->
      // Quaternion
      cv::Mat rotation_matrix;
      cv::Rodrigues(rvec, rotation_matrix);

      tf2::Matrix3x3 tf2_rotation_matrix(
          rotation_matrix.at<double>(0, 0), rotation_matrix.at<double>(0, 1),
          rotation_matrix.at<double>(0, 2), rotation_matrix.at<double>(1, 0),
          rotation_matrix.at<double>(1, 1), rotation_matrix.at<double>(1, 2),
          rotation_matrix.at<double>(2, 0), rotation_matrix.at<double>(2, 1),
          rotation_matrix.at<double>(2, 2));

      tf2::Quaternion tf2_q;
      tf2_rotation_matrix.getRotation(tf2_q);
      armor_msg.pose.orientation = tf2::toMsg(tf2_q);

      // Fill distance to image center (used for target selection strategy)
      armor_msg.distance_to_image_center =
          pnp_solver_->calculateDistanceToCenter(armor.center);

      // Fill visualization markers
      armor_marker_.id++;
      // Small armor: 135mm width approx, Large: 230mm approx
      armor_marker_.scale.y = armor.type == ArmorType::SMALL ? 0.135 : 0.23;
      armor_marker_.pose = armor_msg.pose;

      text_marker_.id++;
      text_marker_.pose.position = armor_msg.pose.position;
      text_marker_.pose.position.y -= 0.1;  // Float text slightly above/below
      text_marker_.text = armor.classfication_result;

      armors_msg_.armors.emplace_back(armor_msg);
      marker_array_.markers.emplace_back(armor_marker_);
      marker_array_.markers.emplace_back(text_marker_);
    } else {
      RCLCPP_WARN(this->get_logger(), "PnP failed!");
    }
  }

  // Publish results
  armors_pub_->publish(armors_msg_);
  publishMarkers();
}

std::unique_ptr<Detector> ArmorDetectorNode::initDetector() {
  // Define parameter descriptors (ranges for integer params)
  rcl_interfaces::msg::ParameterDescriptor param_desc;
  param_desc.integer_range.resize(1);
  param_desc.integer_range[0].step = 1;
  param_desc.integer_range[0].from_value = 0;
  param_desc.integer_range[0].to_value = 255;
  binary_thres_ = declare_parameter("binary_thres", 160, param_desc);

  param_desc.description = "0-RED, 1-BLUE";
  param_desc.integer_range[0].from_value = 0;
  param_desc.integer_range[0].to_value = 1;
  detect_color_ = declare_parameter("detect_color", RED, param_desc);

  Detector::LightParams l_params = {
      .min_ratio = declare_parameter("light.min_ratio", 0.1),
      .max_ratio = declare_parameter("light.max_ratio", 0.4),
      .max_angle = declare_parameter("light.max_angle", 40.0)};

  Detector::ArmorParams a_params = {
      .min_light_ratio = declare_parameter("armor.min_light_ratio", 0.7),
      .min_small_center_distance =
          declare_parameter("armor.min_small_center_distance", 0.8),
      .max_small_center_distance =
          declare_parameter("armor.max_small_center_distance", 3.2),
      .min_large_center_distance =
          declare_parameter("armor.min_large_center_distance", 3.2),
      .max_large_center_distance =
          declare_parameter("armor.max_large_center_distance", 5.5),
      .max_angle = declare_parameter("armor.max_angle", 35.0)};

  // Initialize Neural Network Classifier
  auto pkg_path =
      ament_index_cpp::get_package_share_directory("armor_detector");
  auto model_path = pkg_path + "/model/mlp.onnx";
  auto label_path = pkg_path + "/model/label.txt";
  classifier_threshold_ = this->declare_parameter("classifier_threshold", 0.7);
  std::vector<std::string> ignore_classes = this->declare_parameter(
      "ignore_classes", std::vector<std::string>{"negative"});

  // --- FEATURE: Print all loaded parameters ---
  RCLCPP_INFO(this->get_logger(), "*********** Detector Params ***********");
  RCLCPP_INFO(this->get_logger(), "Binary Thres: %d", binary_thres_);
  RCLCPP_INFO(this->get_logger(), "Detect Color: %d (0-Red, 1-Blue)",
              detect_color_);
  RCLCPP_INFO(this->get_logger(),
              "Light Params: ratio:[%.2f, %.2f], angle:%.2f",
              l_params.min_ratio, l_params.max_ratio, l_params.max_angle);
  RCLCPP_INFO(
      this->get_logger(),
      "Armor Params: min_light_ratio:%.2f, small_dist:[%.2f, %.2f], "
      "large_dist:[%.2f, %.2f], angle:%.2f",
      a_params.min_light_ratio, a_params.min_small_center_distance,
      a_params.max_small_center_distance, a_params.min_large_center_distance,
      a_params.max_large_center_distance, a_params.max_angle);
  RCLCPP_INFO(this->get_logger(), "Classifier Thres: %.2f",
              classifier_threshold_);
  RCLCPP_INFO(this->get_logger(), "***************************************");

  auto detector = std::make_unique<Detector>(binary_thres_, detect_color_,
                                             l_params, a_params);

  detector->classifier = std::make_unique<NumberClassifier>(
      model_path, label_path, classifier_threshold_, ignore_classes);

  return detector;
}

std::vector<Armor> ArmorDetectorNode::detectArmors(
    const sensor_msgs::msg::Image::ConstSharedPtr &img_msg) {
  // Convert ROS Image message to OpenCV Matrix (Zero-Copy)
  auto img = cv_bridge::toCvShare(img_msg, "rgb8")->image;

  // Core Detection Logic
  auto armors = detector_->detect(img);

  // Calculate Latency
  auto final_time = this->now();
  auto latency = (final_time - img_msg->header.stamp).seconds() * 1000;
  RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms");

  // Publish Debug Information
  bool publish_debug = debug_;
  if (publish_debug && debug_publish_rate_ > 0.0) {
    const auto now = this->now();
    const double period = 1.0 / debug_publish_rate_;
    publish_debug = (now - last_debug_pub_time_).seconds() >= period;
    if (publish_debug) {
      last_debug_pub_time_ = now;
    }
  }

  if (publish_debug) {
    binary_img_pub_.publish(
        cv_bridge::CvImage(img_msg->header, "mono8", detector_->binary_img)
            .toImageMsg());

    // Sort lights and armors by X-coordinate for easier debugging/plotting
    std::sort(detector_->debug_lights.data.begin(),
              detector_->debug_lights.data.end(),
              [](const auto &l1, const auto &l2) {
                return l1.center_x < l2.center_x;
              });
    std::sort(detector_->debug_armors.data.begin(),
              detector_->debug_armors.data.end(),
              [](const auto &a1, const auto &a2) {
                return a1.center_x < a2.center_x;
              });

    lights_data_pub_->publish(detector_->debug_lights);
    armors_data_pub_->publish(detector_->debug_armors);

    if (!armors.empty()) {
      auto all_num_img = detector_->getAllNumbersImage();
      number_img_pub_.publish(
          *cv_bridge::CvImage(img_msg->header, "mono8", all_num_img)
               .toImageMsg());
    }

    detector_->drawResults(img);
    // Draw camera center
    cv::circle(img, cam_center_, 5, cv::Scalar(255, 0, 0), 2);
    // Draw latency
    std::stringstream latency_ss;
    latency_ss << "Latency: " << std::fixed << std::setprecision(2) << latency
               << "ms";
    auto latency_s = latency_ss.str();
    cv::putText(img, latency_s, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                1.0, cv::Scalar(0, 255, 0), 2);
    result_img_pub_.publish(
        cv_bridge::CvImage(img_msg->header, "rgb8", img).toImageMsg());
  }

  return armors;
}

void ArmorDetectorNode::createDebugPublishers() {
  lights_data_pub_ =
      this->create_publisher<auto_aim_interfaces::msg::DebugLights>(
          debug_lights_topic_, 10);
  armors_data_pub_ =
      this->create_publisher<auto_aim_interfaces::msg::DebugArmors>(
          debug_armors_topic_, 10);

  binary_img_pub_ =
      image_transport::create_publisher(this, debug_binary_img_topic_);
  number_img_pub_ =
      image_transport::create_publisher(this, debug_number_img_topic_);
  result_img_pub_ =
      image_transport::create_publisher(this, debug_result_img_topic_);
}

void ArmorDetectorNode::destroyDebugPublishers() {
  lights_data_pub_.reset();
  armors_data_pub_.reset();

  binary_img_pub_.shutdown();
  number_img_pub_.shutdown();
  result_img_pub_.shutdown();
}

void ArmorDetectorNode::publishMarkers() {
  using Marker = visualization_msgs::msg::Marker;
  // If no armor detected, DELETE the marker. Otherwise ADD/MODIFY it.
  armor_marker_.action =
      armors_msg_.armors.empty() ? Marker::DELETE : Marker::ADD;

  // Bug fix: Always push back markers even if deleting, so Rviz knows what ID to
  // delete
  marker_array_.markers.emplace_back(armor_marker_);
  marker_pub_->publish(marker_array_);
}

rcl_interfaces::msg::SetParametersResult ArmorDetectorNode::onSetParameters(
    const std::vector<rclcpp::Parameter> &params) {
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  for (const auto &param : params) {
    if (param.get_name() == "debug") {
      const bool new_debug = param.as_bool();
      if (new_debug != debug_) {
        debug_ = new_debug;
        if (debug_) {
          createDebugPublishers();
          last_debug_pub_time_ = this->now();
        } else {
          destroyDebugPublishers();
        }
      }
    } else if (param.get_name() == "binary_thres") {
      const int value = param.as_int();
      if (value < 0 || value > 255) {
        result.successful = false;
        result.reason = "binary_thres must be in [0, 255]";
        return result;
      }
      binary_thres_ = value;
      detector_->binary_thres = value;
    } else if (param.get_name() == "detect_color") {
      const int value = param.as_int();
      if (value != RED && value != BLUE) {
        result.successful = false;
        result.reason = "detect_color must be 0 (RED) or 1 (BLUE)";
        return result;
      }
      detect_color_ = value;
      detector_->detect_color = value;
    } else if (param.get_name() == "classifier_threshold") {
      const double value = param.as_double();
      if (value < 0.0 || value > 1.0) {
        result.successful = false;
        result.reason = "classifier_threshold must be in [0.0, 1.0]";
        return result;
      }
      classifier_threshold_ = value;
      detector_->classifier->threshold = value;
    } else if (param.get_name() == "debug_publish_rate") {
      const double value = param.as_double();
      if (value < 0.0) {
        result.successful = false;
        result.reason = "debug_publish_rate must be >= 0.0";
        return result;
      }
      debug_publish_rate_ = value;
    }
  }

  return result;
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorDetectorNode)
