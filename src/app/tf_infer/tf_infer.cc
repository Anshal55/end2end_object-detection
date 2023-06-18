#include <chrono>
#include <vector>

#include "src/includes/image/image_helper.h"
#include "src/includes/object_detection/bounding_box.h"
#include "src/includes/object_detection/classifier_helper.h"
#include "src/includes/object_detection/non_max_supression.h"
#include "src/includes/object_detection/object_detection_helper.h"

/**
 * Main Function that initializes the camera and performs
 * Object detection and Classification on it.
 */
int main(int argc, const char **argv) {
  // Initialize logging
  google::InitGoogleLogging(argv[0]);

  // Initialize ImageHelper
  ImageHelper img_helper;

  // Define ObjectDetectionHelper
  ObjectDetectionHelper object_detector;

  // Define ClassifierHelper
  ClassifierHelper classifier;

  cv::VideoCapture cap;

  if (argc > 1 && std::string(argv[1]) == "rtsp") {
    // Open RTSP stream
    std::string rtsp_url = "rtsp://admin:admin@192.168.0.100:1935";
    cap.open(rtsp_url);
  } else {
    // Open webcam
    cap.open(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
  }

  if (!cap.isOpened()) {
    LOG(FATAL) << "Failed to open video source";
  }

  // Initialize FPS calculation
  int frame_count = 0;
  auto start_time = std::chrono::high_resolution_clock::now();

  // Create matrices to hold the converted and resized images
  cv::Mat rgb_image;
  cv::Mat bgr_image;

  while (true) {
    cap >> bgr_image;
    if (bgr_image.empty()) {
      std::cerr << "Failed to read frame from webcam\n";
      return 1;
    }

    // Pre-process image for Object Detection
    auto img_tensor = img_helper.PreProcessImages(bgr_image, rgb_image, 416);

    // Perform Object Detection inference
    object_detector.UpdateTensor(img_tensor);
    bool has_detected = object_detector.RunInference();

    if (!has_detected) {
      continue;
    }

    // Gather bounding boxes from the detected objects
    auto bounding_boxes = object_detector.GatherBoundingBoxes(0.4);
    LOG(INFO) << "Number of detections = " << bounding_boxes.size();

    // Apply Non-Maximum Suppression
    auto filtered_bbs = NonMaximumSuppression(bounding_boxes, 0.5);
    LOG(INFO) << "After Non-Maximum Suppression = " << filtered_bbs.size();

    // Iterate over the filtered bounding boxes and draw them if classifier
    // predictions are true
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

      // Update classifier with the region of interest
      classifier.UpdateTensor(bgr_image, x1, y1, x2, y2);
      bool is_hand = classifier.RunInference();

      LOG(INFO) << "Is it a Hand? = " << is_hand;

      if (!is_hand)
        continue;

      // Draw bounding box
      img_helper.DrawOnBBs(bgr_image, x1, y1, x2, y2, score);
    }

    // Calculate FPS
    frame_count++;
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              end_time - start_time)
                              .count() /
                          1000.0;
    double fps = frame_count / elapsed_time;

    // Draw the FPS
    img_helper.DrawFps(bgr_image, fps);

    // Show the bgr_image
    cv::imshow("Detections", bgr_image);

    // Check if the user pressed the 'q' key to quit
    if (cv::waitKey(1) == 'q') {
      break;
    }
  }

  // Release resources
  cap.release();
  cv::destroyAllWindows();

  return 0;
}