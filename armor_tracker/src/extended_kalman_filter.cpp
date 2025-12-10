// Copyright 2022 Chen Jun
// Copyright 2025 Yihan Li
// Licensed under the MIT License.

#include "armor_tracker/extended_kalman_filter.hpp"

namespace rm_auto_aim {

/**
 * @brief Construct a new Extended Kalman Filter object.
 * Initializes the function pointers and initial covariance matrices.
 *
 * @param f State transition function: x_k = f(x_{k-1})
 * @param h Observation function: z_k = h(x_k)
 * @param j_f Jacobian of state transition function f w.r.t state x
 * @param j_h Jacobian of observation function h w.r.t state x
 * @param u_q Function to update process noise covariance matrix Q
 * @param u_r Function to update measurement noise covariance matrix R
 * @param P0 Initial state covariance matrix
 */
ExtendedKalmanFilter::ExtendedKalmanFilter(const VecVecFunc &f,
                                           const VecVecFunc &h,
                                           const VecMatFunc &j_f,
                                           const VecMatFunc &j_h,
                                           const VoidMatFunc &u_q,
                                           const VecMatFunc &u_r,
                                           const Eigen::MatrixXd &P0)
    : f(f),
      h(h),
      jacobian_f(j_f),
      jacobian_h(j_h),
      update_Q(u_q),
      update_R(u_r),
      P_post(P0),
      n(P0.rows()),
      I(Eigen::MatrixXd::Identity(n, n)),
      x_pri(n),
      x_post(n) {}

/**
 * @brief Set the initial state vector x0.
 *
 * @param x0 Initial state vector (e.g., [x, y, z, vx, vy, vz, ...])
 */
void ExtendedKalmanFilter::setState(const Eigen::VectorXd &x0) {
  x_post = x0;
}

/**
 * @brief Predicts the next state.
 * Step 1: Predict state using the nonlinear function f.
 * Step 2: Predict covariance using the Jacobian F.
 *
 * @return Eigen::MatrixXd The predicted state vector (x_pri).
 */
Eigen::MatrixXd ExtendedKalmanFilter::predict() {
  // Calculate Jacobian matrix F at the current posterior state.
  F = jacobian_f(x_post);
  // Calculate process noise covariance matrix Q.
  Q = update_Q();

  // 1. State Prediction (A Priori State)
  // x_{k|k-1} = f(x_{k-1|k-1})
  x_pri = f(x_post);

  // 2. Covariance Prediction (A Priori Covariance)
  // P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
  P_pri = F * P_post * F.transpose() + Q;

  // Update internal state to a priori.
  // This handles the case where update() might not be called (e.g., no measurement).
  // If update() is called later, x_post and P_post will be corrected.
  x_post = x_pri;
  P_post = P_pri;

  return x_pri;
}

/**
 * @brief Updates the state based on a new measurement.
 * Step 1: Calculate Kalman Gain K.
 * Step 2: Update state estimate using measurement residual.
 * Step 3: Update covariance estimate.
 *
 * @param z The measurement vector (observation).
 * @return Eigen::MatrixXd The corrected state vector (x_post).
 */
Eigen::MatrixXd ExtendedKalmanFilter::update(const Eigen::VectorXd &z) {
  // Calculate Jacobian matrix H at the predicted state.
  H = jacobian_h(x_pri);
  // Calculate measurement noise covariance matrix R based on the measurement.
  R = update_R(z);

  // 1. Calculate Kalman Gain
  // K = P_pri * H^T * (H * P_pri * H^T + R)^-1
  // (H * P_pri * H^T + R) is often called the Innovation Covariance (S).
  K = P_pri * H.transpose() * (H * P_pri * H.transpose() + R).inverse();

  // 2. Update State Estimate (A Posteriori State)
  // x_{k|k} = x_{k|k-1} + K * (z - h(x_{k|k-1}))
  // (z - h(x)) is the measurement residual (innovation).
  x_post = x_pri + K * (z - h(x_pri));

  // 3. Update Covariance Estimate (A Posteriori Covariance)
  // P_{k|k} = (I - K * H) * P_{k|k-1}
  P_post = (I - K * H) * P_pri;

  return x_post;
}

}  // namespace rm_auto_aim