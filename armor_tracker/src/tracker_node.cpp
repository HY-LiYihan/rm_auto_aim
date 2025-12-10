// Copyright 2022 Chen Jun
// Copyright 2025 Yihan Li
// Licensed under the MIT License.

#include "armor_tracker/tracker_node.hpp"

// STD
#include <memory>
#include <vector>

namespace rm_auto_aim {

/**
 * @brief Construct a new ArmorTrackerNode.
 * Initializes subscribers, publishers, TF listener, and the Tracker core logic.
 *
 * @param options ROS 2 Node options.
 */
ArmorTrackerNode::ArmorTrackerNode(const rclcpp::NodeOptions &options)
    : Node("armor_tracker", options) {
  RCLCPP_INFO(this->get_logger(), "Starting TrackerNode!");

  // --- Parameters Declaration ---
  // Maximum allowable armor distance in the XOY plane (filter out outliers).
  max_armor_distance_ = this->declare_parameter("max_armor_distance", 10.0);

  // Tracker Parameters
  double max_match_distance =
      this->declare_parameter("tracker.max_match_distance", 0.15);
  double max_match_yaw_diff =
      this->declare_parameter("tracker.max_match_yaw_diff", 1.0);
  
  // Initialize the Tracker core logic class
  tracker_ =
      std::make_unique<Tracker>(max_match_distance, max_match_yaw_diff);
  tracker_->tracking_thres = this->declare_parameter("tracker.tracking_thres", 5);
  lost_time_thres_ = this->declare_parameter("tracker.lost_time_thres", 0.3);

  // --- EKF (Extended Kalman Filter) Configuration ---
  // State Vector (9 dims): [xc, v_xc, yc, v_yc, za, v_za, yaw, v_yaw, r]
  // Measurement Vector (4 dims): [xa, ya, za, yaw]

  // 1. Process Function f(x): x_k = f(x_{k-1})
  // Assuming Constant Velocity (CV) model for linear motion and yaw.
  auto f = [this](const Eigen::VectorXd &x) {
    Eigen::VectorXd x_new = x;
    x_new(0) += x(1) * dt_;  // xc += v_xc * dt
    x_new(2) += x(3) * dt_;  // yc += v_yc * dt
    x_new(4) += x(5) * dt_;  // za += v_za * dt
    x_new(6) += x(7) * dt_;  // yaw += v_yaw * dt
    return x_new;
  };

  // 2. Jacobian of Process Function J_f
  // Since f(x) is linear (CV model), the Jacobian is constant.
  auto j_f = [this](const Eigen::VectorXd &) {
    Eigen::MatrixXd f(9, 9);
    // clang-format off
    f <<  1,   dt_, 0,   0,   0,   0,   0,   0,   0,
          0,   1,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   1,   dt_, 0,   0,   0,   0,   0, 
          0,   0,   0,   1,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   1,   dt_, 0,   0,   0,
          0,   0,   0,   0,   0,   1,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   1,   dt_, 0,
          0,   0,   0,   0,   0,   0,   0,   1,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   1;
    // clang-format on
    return f;
  };

  // 3. Observation Function h(x): z_k = h(x_k)
  // Maps the vehicle center state to the armor position.
  // xa = xc - r * cos(yaw)
  // ya = yc - r * sin(yaw)
  auto h = [](const Eigen::VectorXd &x) {
    Eigen::VectorXd z(4);
    double xc = x(0), yc = x(2), yaw = x(6), r = x(8);
    z(0) = xc - r * cos(yaw);  // xa
    z(1) = yc - r * sin(yaw);  // ya
    z(2) = x(4);               // za
    z(3) = x(6);               // yaw
    return z;
  };

  // 4. Jacobian of Observation Function J_h
  auto j_h = [](const Eigen::VectorXd &x) {
    Eigen::MatrixXd h(4, 9);
    double yaw = x(6), r = x(8);
    // clang-format off
    //    xc   v_xc yc   v_yc za   v_za yaw         v_yaw r
    h <<  1,   0,   0,   0,   0,   0,   r*sin(yaw), 0,   -cos(yaw),
          0,   0,   1,   0,   0,   0,   -r*cos(yaw),0,   -sin(yaw),
          0,   0,   0,   0,   1,   0,   0,          0,   0,
          0,   0,   0,   0,   0,   0,   1,          0,   0;
    // clang-format on
    return h;
  };

  // 5. Process Noise Covariance Matrix Q
  // Represents uncertainty in the process model (e.g. acceleration, friction).
  s2qxyz_ = declare_parameter("ekf.sigma2_q_xyz", 20.0);
  s2qyaw_ = declare_parameter("ekf.sigma2_q_yaw", 100.0);
  s2qr_ = declare_parameter("ekf.sigma2_q_r", 800.0);

  auto u_q = [this]() {
    Eigen::MatrixXd q(9, 9);
    double t = dt_, x = s2qxyz_, y = s2qyaw_, r = s2qr_;
    // Standard CV model Q matrix formulas
    double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
    double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * x, q_vy_vy = pow(t, 2) * y;
    double q_r = pow(t, 4) / 4 * r;
    
    // clang-format off
    //    xc      v_xc    yc      v_yc    za      v_za    yaw     v_yaw   r
    q <<  q_x_x,  q_x_vx, 0,      0,      0,      0,      0,      0,      0,
          q_x_vx, q_vx_vx,0,      0,      0,      0,      0,      0,      0,
          0,      0,      q_x_x,  q_x_vx, 0,      0,      0,      0,      0,
          0,      0,      q_x_vx, q_vx_vx,0,      0,      0,      0,      0,
          0,      0,      0,      0,      q_x_x,  q_x_vx, 0,      0,      0,
          0,      0,      0,      0,      q_x_vx, q_vx_vx,0,      0,      0,
          0,      0,      0,      0,      0,      0,      q_y_y,  q_y_vy, 0,
          0,      0,      0,      0,      0,      0,      q_y_vy, q_vy_vy,0,
          0,      0,      0,      0,      0,      0,      0,      0,      q_r;
    // clang-format on
    return q;
  };

  // 6. Measurement Noise Covariance Matrix R
  // Represents uncertainty in sensor measurements (camera noise).
  r_xyz_factor = declare_parameter("ekf.r_xyz_factor", 0.05);
  r_yaw = declare_parameter("ekf.r_yaw", 0.02);
  
  auto u_r = [this](const Eigen::VectorXd &z) {
    Eigen::DiagonalMatrix<double, 4> r;
    double x = r_xyz_factor;
    // R is proportional to distance: larger distance => larger noise
    r.diagonal() << abs(x * z[0]), abs(x * z[1]), abs(x * z[2]), r_yaw;
    return r;
  };

  // 7. Initial Error Covariance Matrix P0
  Eigen::DiagonalMatrix<double, 9> p0;
  p0.setIdentity();
  
  // Initialize EKF with all the above functions
  tracker_->ekf = ExtendedKalmanFilter{f, h, j_f, j_h, u_q, u_r, p0};

  // --- Reset Service ---
  using std::placeholders::_1;
  using std::placeholders::_2;
  reset_tracker_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/tracker/reset",
      [this](const std_srvs::srv::Trigger::Request::SharedPtr,
             std_srvs::srv::Trigger::Response::SharedPtr response) {
        tracker_->tracker_state = Tracker::LOST;
        response->success = true;
        RCLCPP_INFO(this->get_logger(), "Tracker reset!");
        return;
      });

  // --- TF2 & Subscriber Setup ---
  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
      this->get_node_base_interface(), this->get_node_timers_interface());
  tf2_buffer_->setCreateTimerInterface(timer_interface);
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

  // Subscriber with MessageFilter to sync with TF
  armors_sub_.subscribe(this, "/detector/armors", rmw_qos_profile_sensor_data);
  target_frame_ = this->declare_parameter("target_frame", "odom");
  tf2_filter_ = std::make_shared<tf2_filter>(
      armors_sub_, *tf2_buffer_, target_frame_, 10,
      this->get_node_logging_interface(), this->get_node_clock_interface(),
      std::chrono::duration<int>(1));
  
  // Callback when TF is available
  tf2_filter_->registerCallback(&ArmorTrackerNode::armorsCallback, this);

  // --- Publishers Setup ---
  info_pub_ = this->create_publisher<auto_aim_interfaces::msg::TrackerInfo>(
      "/tracker/info", 10);
  target_pub_ = this->create_publisher<auto_aim_interfaces::msg::Target>(
      "/tracker/target", rclcpp::SensorDataQoS());
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/tracker/marker", 10);

  // Init visualization markers config
  // ... (omitted marker setup code for brevity as it's standard) ...
  // [Marker setup code from original snippet remains here]
  position_marker_.ns = "position";
  position_marker_.type = visualization_msgs::msg::Marker::SPHERE;
  position_marker_.scale.x = position_marker_.scale.y = position_marker_.scale.z = 0.1;
  position_marker_.color.a = 1.0;
  position_marker_.color.g = 1.0;
  linear_v_marker_.type = visualization_msgs::msg::Marker::ARROW;
  linear_v_marker_.ns = "linear_v";
  linear_v_marker_.scale.x = 0.03;
  linear_v_marker_.scale.y = 0.05;
  linear_v_marker_.color.a = 1.0;
  linear_v_marker_.color.r = 1.0;
  linear_v_marker_.color.g = 1.0;
  angular_v_marker_.type = visualization_msgs::msg::Marker::ARROW;
  angular_v_marker_.ns = "angular_v";
  angular_v_marker_.scale.x = 0.03;
  angular_v_marker_.scale.y = 0.05;
  angular_v_marker_.color.a = 1.0;
  angular_v_marker_.color.b = 1.0;
  angular_v_marker_.color.g = 1.0;
  armor_marker_.ns = "armors";
  armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
  armor_marker_.scale.x = 0.03;
  armor_marker_.scale.z = 0.125;
  armor_marker_.color.a = 1.0;
  armor_marker_.color.r = 1.0;
}

void ArmorTrackerNode::armorsCallback(
    const auto_aim_interfaces::msg::Armors::SharedPtr armors_msg) {
  // 1. Transform armor position from Image Frame to World Frame (odom/map)
  for (auto &armor : armors_msg->armors) {
    geometry_msgs::msg::PoseStamped ps;
    ps.header = armors_msg->header;
    ps.pose = armor.pose;
    try {
      armor.pose = tf2_buffer_->transform(ps, target_frame_).pose;
    } catch (const tf2::ExtrapolationException &ex) {
      RCLCPP_ERROR(get_logger(), "Error while transforming %s", ex.what());
      return;
    }
  }

  // 2. Filter abnormal armors (e.g., too high or too far)
  armors_msg->armors.erase(
      std::remove_if(
          armors_msg->armors.begin(), armors_msg->armors.end(),
          [this](const auto_aim_interfaces::msg::Armor &armor) {
            return abs(armor.pose.position.z) > 1.2 ||
                   Eigen::Vector2d(armor.pose.position.x, armor.pose.position.y)
                           .norm() > max_armor_distance_;
          }),
      armors_msg->armors.end());

  // 3. Prepare result messages
  auto_aim_interfaces::msg::TrackerInfo info_msg;
  auto_aim_interfaces::msg::Target target_msg;
  rclcpp::Time time = armors_msg->header.stamp;
  target_msg.header.stamp = time;
  target_msg.header.frame_id = target_frame_;

  // 4. Update Tracker State Machine
  if (tracker_->tracker_state == Tracker::LOST) {
    tracker_->init(armors_msg);
    target_msg.tracking = false;
  } else {
    // Calculate dt (time interval) for Kalman Filter
    dt_ = (time - last_time_).seconds();
    tracker_->lost_thres = static_cast<int>(lost_time_thres_ / dt_);
    
    // Core Update Logic
    tracker_->update(armors_msg);

    // Publish Debug Info
    info_msg.position_diff = tracker_->info_position_diff;
    info_msg.yaw_diff = tracker_->info_yaw_diff;
    info_msg.position.x = tracker_->measurement(0);
    info_msg.position.y = tracker_->measurement(1);
    info_msg.position.z = tracker_->measurement(2);
    info_msg.yaw = tracker_->measurement(3);
    info_pub_->publish(info_msg);

    // Fill Target Message if Tracking
    if (tracker_->tracker_state == Tracker::DETECTING) {
      target_msg.tracking = false;
    } else if (tracker_->tracker_state == Tracker::TRACKING ||
               tracker_->tracker_state == Tracker::TEMP_LOST) {
      target_msg.tracking = true;
      const auto &state = tracker_->target_state;
      target_msg.id = tracker_->tracked_id;
      target_msg.armors_num = static_cast<int>(tracker_->tracked_armors_num);
      target_msg.position.x = state(0);
      target_msg.velocity.x = state(1);
      target_msg.position.y = state(2);
      target_msg.velocity.y = state(3);
      target_msg.position.z = state(4);
      target_msg.velocity.z = state(5);
      target_msg.yaw = state(6);
      target_msg.v_yaw = state(7);
      target_msg.radius_1 = state(8);
      target_msg.radius_2 = tracker_->another_r;
      target_msg.dz = tracker_->dz;
    }
  }

  last_time_ = time;

  target_pub_->publish(target_msg);

  publishMarkers(target_msg);
}

void ArmorTrackerNode::publishMarkers(
    const auto_aim_interfaces::msg::Target &target_msg) {
  // Sync headers
  position_marker_.header = target_msg.header;
  linear_v_marker_.header = target_msg.header;
  angular_v_marker_.header = target_msg.header;
  armor_marker_.header = target_msg.header;

  visualization_msgs::msg::MarkerArray marker_array;
  
  if (target_msg.tracking) {
    // Unpack target data
    double yaw = target_msg.yaw, r1 = target_msg.radius_1, r2 = target_msg.radius_2;
    double xc = target_msg.position.x, yc = target_msg.position.y, za = target_msg.position.z;
    double vx = target_msg.velocity.x, vy = target_msg.velocity.y, vz = target_msg.velocity.z;
    double dz = target_msg.dz;

    // 1. Position Marker (Sphere)
    position_marker_.action = visualization_msgs::msg::Marker::ADD;
    position_marker_.pose.position.x = xc;
    position_marker_.pose.position.y = yc;
    position_marker_.pose.position.z = za + dz / 2;

    // 2. Linear Velocity Marker (Arrow)
    linear_v_marker_.action = visualization_msgs::msg::Marker::ADD;
    linear_v_marker_.points.clear();
    linear_v_marker_.points.emplace_back(position_marker_.pose.position);
    geometry_msgs::msg::Point arrow_end = position_marker_.pose.position;
    arrow_end.x += vx;
    arrow_end.y += vy;
    arrow_end.z += vz;
    linear_v_marker_.points.emplace_back(arrow_end);

    // 3. Angular Velocity Marker (Arrow pointing up)
    angular_v_marker_.action = visualization_msgs::msg::Marker::ADD;
    angular_v_marker_.points.clear();
    angular_v_marker_.points.emplace_back(position_marker_.pose.position);
    arrow_end = position_marker_.pose.position;
    arrow_end.z += target_msg.v_yaw / M_PI; // Visualize rotation speed as arrow height
    angular_v_marker_.points.emplace_back(arrow_end);

    // 4. Armor Markers (Cubes rotating around center)
    armor_marker_.action = visualization_msgs::msg::Marker::ADD;
    armor_marker_.scale.y = tracker_->tracked_armor.type == "small" ? 0.135 : 0.23;
    bool is_current_pair = true;
    size_t a_n = target_msg.armors_num;
    geometry_msgs::msg::Point p_a;
    double r = 0;
    
    // Loop to draw all 4 (or 2/3) armors based on current yaw state
    for (size_t i = 0; i < a_n; i++) {
      double tmp_yaw = yaw + i * (2 * M_PI / a_n);
      
      // Handle radius and height difference for 4-armor layout
      if (a_n == 4) {
        r = is_current_pair ? r1 : r2;
        p_a.z = za + (is_current_pair ? 0 : dz);
        is_current_pair = !is_current_pair;
      } else {
        r = r1;
        p_a.z = za;
      }
      p_a.x = xc - r * cos(tmp_yaw);
      p_a.y = yc - r * sin(tmp_yaw);

      armor_marker_.id = i;
      armor_marker_.pose.position = p_a;
      tf2::Quaternion q;
      q.setRPY(0, target_msg.id == "outpost" ? -0.26 : 0.26, tmp_yaw);
      armor_marker_.pose.orientation = tf2::toMsg(q);
      marker_array.markers.emplace_back(armor_marker_);
    }
  } else {
    // If lost, delete all markers
    position_marker_.action = visualization_msgs::msg::Marker::DELETE;
    linear_v_marker_.action = visualization_msgs::msg::Marker::DELETE;
    angular_v_marker_.action = visualization_msgs::msg::Marker::DELETE;
    armor_marker_.action = visualization_msgs::msg::Marker::DELETE;
    marker_array.markers.emplace_back(armor_marker_);
  }

  marker_array.markers.emplace_back(position_marker_);
  marker_array.markers.emplace_back(linear_v_marker_);
  marker_array.markers.emplace_back(angular_v_marker_);
  marker_pub_->publish(marker_array);
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorTrackerNode)