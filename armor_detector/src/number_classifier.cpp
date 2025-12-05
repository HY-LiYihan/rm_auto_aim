// Copyright 2022 Chen Jun
// Copyright 2025 Yihan Li
// Licensed under the MIT License.

#include "armor_detector/number_classifier.hpp"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// STL
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"

namespace rm_auto_aim {

/**
 * @brief Construct a new Number Classifier object.
 * Loads the ONNX model and the label file.
 *
 * @param model_path Path to the .onnx model file.
 * @param label_path Path to the label.txt file.
 * @param thre Threshold for classification confidence.
 * @param ignore_classes List of classes to ignore (e.g., "negative").
 */
NumberClassifier::NumberClassifier(
    const std::string &model_path, const std::string &label_path,
    const double thre, const std::vector<std::string> &ignore_classes)
    : threshold(thre), ignore_classes_(ignore_classes) {
  // Load the neural network model using OpenCV DNN module.
  net_ = cv::dnn::readNetFromONNX(model_path);

  // Load class labels from the text file.
  std::ifstream label_file(label_path);
  std::string line;
  while (std::getline(label_file, line)) {
    class_names_.push_back(line);
  }
}

/**
 * @brief Extracts the number ROI from the armor image using Perspective Transform.
 *
 * @param src The source raw image.
 * @param armors The vector of detected armors to process.
 */
void NumberClassifier::extractNumbers(const cv::Mat &src,
                                      std::vector<Armor> &armors) {
  // Constants for the perspective transform target size.
  // We want to normalize the armor image to a standard size for the CNN.
  const int light_length = 12;
  const int warp_height = 28;
  const int small_armor_width = 32;
  const int large_armor_width = 54;
  
  // The final ROI size fed into the network.
  const cv::Size roi_size(20, 28);

  for (auto &armor : armors) {
    // 1. Define source points (vertices of the lights in the original image).
    // Order: Bottom-Left -> Top-Left -> Top-Right -> Bottom-Right
    cv::Point2f lights_vertices[4] = {
        armor.left_light.bottom, armor.left_light.top, armor.right_light.top,
        armor.right_light.bottom};

    // 2. Define target points (where we want the lights to be in the warped image).
    // This effectively "straightens" the armor plate.
    const int top_light_y = (warp_height - light_length) / 2 - 1;
    const int bottom_light_y = top_light_y + light_length;
    
    // Select width based on armor type.
    const int warp_width =
        armor.type == ArmorType::SMALL ? small_armor_width : large_armor_width;

    cv::Point2f target_vertices[4] = {
        cv::Point(0, bottom_light_y),
        cv::Point(0, top_light_y),
        cv::Point(warp_width - 1, top_light_y),
        cv::Point(warp_width - 1, bottom_light_y),
    };

    // 3. Perform Perspective Transform (Warp).
    cv::Mat number_image;
    auto rotation_matrix =
        cv::getPerspectiveTransform(lights_vertices, target_vertices);
    cv::warpPerspective(src, number_image, rotation_matrix,
                        cv::Size(warp_width, warp_height));

    // 4. Crop the center ROI.
    // The previous warp might include some margin; we only want the center number.
    number_image = number_image(
        cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));

    // 5. Binarize the image.
    // Convert to Grayscale -> Otsu's Thresholding.
    // This helps in highlighting the number shape against the background.
    cv::cvtColor(number_image, number_image, cv::COLOR_RGB2GRAY);
    cv::threshold(number_image, number_image, 0, 255,
                  cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Store the processed image in the armor object.
    armor.number_img = number_image;
  }
}

/**
 * @brief Classifies the extracted number images using the loaded neural network.
 * Filters out low-confidence results and mismatches.
 *
 * @param armors The vector of armors to classify.
 */
void NumberClassifier::classify(std::vector<Armor> &armors) {
  for (auto &armor : armors) {
    cv::Mat image = armor.number_img.clone();

    // 1. Normalize the image (0-255 -> 0.0-1.0).
    // Neural networks work better with normalized inputs.
    image = image / 255.0;

    // 2. Create a 4-dimensional blob from the image.
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob);

    // 3. Inference
    net_.setInput(blob);
    cv::Mat outputs = net_.forward();

    // 4. Softmax calculation (Convert logits to probabilities).
    // Find max value to prevent overflow during exp().
    float max_prob =
        *std::max_element(outputs.begin<float>(), outputs.end<float>());
    cv::Mat softmax_prob;
    cv::exp(outputs - max_prob, softmax_prob);
    float sum = static_cast<float>(cv::sum(softmax_prob)[0]);
    softmax_prob /= sum;

    // 5. Get the best class label and confidence.
    double confidence;
    cv::Point class_id_point;
    minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr,
              &class_id_point);
    int label_id = class_id_point.x;

    armor.confidence = confidence;
    armor.number = class_names_[label_id];

    // Format the result string for debugging (e.g., "1: 95.2%").
    std::stringstream result_ss;
    result_ss << armor.number << ": " << std::fixed << std::setprecision(1)
              << armor.confidence * 100.0 << "%";
    armor.classfication_result = result_ss.str();
  }

  // 6. Filter results.
  // Remove armors that don't meet criteria.
  armors.erase(
      std::remove_if(
          armors.begin(), armors.end(),
          [this](const Armor &armor) {
            // Criteria A: Confidence threshold.
            if (armor.confidence < threshold) {
              return true;
            }

            // Criteria B: Ignore classes (e.g., "negative" samples).
            for (const auto &ignore_class : ignore_classes_) {
              if (armor.number == ignore_class) {
                return true;
              }
            }

            // Criteria C: Mismatch between geometric type and recognized number.
            // Example: "1" is almost always a Large armor (Hero).
            // Note: These rules depend on the specific RM season rules.
            bool mismatch_armor_type = false;
            if (armor.type == ArmorType::LARGE) {
              // Large armor should typically be "1", "Base", "Sentry".
              // If it's recognized as "2" or "Outpost" or "Guard", it's likely wrong.
              mismatch_armor_type = armor.number == "outpost" ||
                                    armor.number == "2" ||
                                    armor.number == "guard";
            } else if (armor.type == ArmorType::SMALL) {
              // Small armor should typically be "2", "3", "4", "5".
              // If recognized as "1" or "Base", it's likely wrong.
              mismatch_armor_type =
                  armor.number == "1" || armor.number == "base";
            }
            return mismatch_armor_type;
          }),
      armors.end());
}

}  // namespace rm_auto_aim