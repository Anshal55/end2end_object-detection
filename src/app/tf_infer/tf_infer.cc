#include <chrono>
#include <iostream>
#include <vector>

// GLOG
#include "glog/logging.h"
#include "glog/stl_logging.h"

// opencv dependencies
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

// tflite dependencies
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include "src/includes/object_detection/bounding_box.h"
#include "src/includes/object_detection/classifier_helper.h"
#include "src/includes/object_detection/non_max_supression.h"
#include "src/includes/object_detection/object_detection_helper.h"

int main(int argc, const char **argv) {
  google::InitGoogleLogging(argv[0]);

  // Define object detection helper
  ObjectDetectionHelper object_detector;

  // Define classifier helper
  ClassifierHelper classifier;

  // Open webcam
  cv::VideoCapture cap(0);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  cap.set(cv::CAP_PROP_FPS, 30);
  if (!cap.isOpened()) {
    LOG(FATAL) << "Failed to open webcam";
  }

  // Initialize FPS calculation
  int frame_count = 0;
  auto start_time = std::chrono::high_resolution_clock::now();

  // Create matrices to hold the converted and resized images
  cv::Mat rgb_image;
  cv::Mat img_tensor;

  // Read frame from webcam
  cv::Mat bgr_image;

  while (true) {
    cap >> bgr_image;
    if (bgr_image.empty()) {
      std::cerr << "Failed to read frame from webcam\n";
      return 1;
    }

    // Convert image from BGR to RGB color space
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

    // Resize the image to (416, 416)
    cv::resize(rgb_image, rgb_image, cv::Size(416, 416));
    // cv::resize(rgb_image, rgb_image, cv::Size(300, 300));

    // Convert the image to a tensor of shape (416, 416, 3)
    std::vector<cv::Mat> channels;
    cv::split(rgb_image, channels);
    cv::merge(channels, img_tensor);
    img_tensor.convertTo(img_tensor, CV_32FC3);
    img_tensor /= 255.0;

    // note: Inference for Object detection here
    object_detector.UpdateTensor(img_tensor);
    bool has_detected = object_detector.RunInference();

    if (!has_detected) {
      continue;
    }

    auto bounding_boxes = object_detector.GatherBoundingBoxes(0.4);
    LOG(INFO) << "Number of detections = " << bounding_boxes.size();

    // Apply Non Max supression
    auto filtered_bbs = NonMaximumSuppression(bounding_boxes, 0.5);

    LOG(INFO) << "After NonMaxSupression = " << filtered_bbs.size();

    for (const auto &bb : filtered_bbs) {
      // Get detection data
      float score = bb.confidence;
      float ymin = bb.ymin;
      float xmin = bb.xmin;
      float ymax = bb.ymax;
      float xmax = bb.xmax;

      // Convert box coordinates to pixel coordinates
      int x1 = static_cast<int>(xmin * bgr_image.cols);
      int y1 = static_cast<int>(ymin * bgr_image.rows);
      int x2 = static_cast<int>(xmax * bgr_image.cols);
      int y2 = static_cast<int>(ymax * bgr_image.rows);

      // note: Add the classifier model here to verify the output
      classifier.UpdateTensor(bgr_image, x1, y1, x2, y2);
      bool is_hand = classifier.RunInference();

      LOG(INFO) << "Is it a Hand = " << is_hand;

      if (!is_hand)
        continue;

      // Draw bounding box on the bgr_image
      cv::rectangle(bgr_image, cv::Point(x1, y1), cv::Point(x2, y2),
                    cv::Scalar(0, 255, 0), 2);

      // Draw label on the bgr_image
      std::string label_text = "Score: " + std::to_string(score);
      int baseline;
      cv::Size label_size = cv::getTextSize(
          label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
      cv::rectangle(
          bgr_image, cv::Point(x1, y1 - label_size.height - baseline - 10),
          cv::Point(x1 + label_size.width, y1), cv::Scalar(255, 255, 255), -1);
      cv::putText(bgr_image, label_text, cv::Point(x1, y1 - baseline - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    // Calculate FPS
    frame_count++;
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              end_time - start_time)
                              .count() /
                          1000.0;
    double fps = frame_count / elapsed_time;

    // Draw FPS on the image
    std::string fps_text = "FPS: " + std::to_string((int)fps);
    int baseline;
    cv::Size fps_size =
        cv::getTextSize(fps_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::rectangle(
        bgr_image, cv::Point(10, 10),
        cv::Point(10 + fps_size.width, 10 + fps_size.height + baseline),
        cv::Scalar(255, 255, 255), -1);
    cv::putText(bgr_image, fps_text, cv::Point(10, 10 + fps_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

    // Show the bgr_image
    cv::imshow("Detections", bgr_image);

    // Check if the user pressed the 'q' key
    if (cv::waitKey(1) == 'q') {
      break;
    }
  }

  // Release resources
  cap.release();
  cv::destroyAllWindows();

  return 0;
}