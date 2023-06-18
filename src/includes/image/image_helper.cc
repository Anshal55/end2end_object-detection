#include "src/includes/image/image_helper.h"

ImageHelper::ImageHelper() {
  // Constructor: Initialize the ImageHelper
  LOG(INFO) << "ImageHelper initialized.";
}

ImageHelper::~ImageHelper() {
  // Destructor: Close the ImageHelper
  LOG(INFO) << "Closing ImageHelper.";
}

cv::Mat ImageHelper::PreProcessImages(cv::Mat &bgr_image, cv::Mat &rgb_image,
                                      int img_shape) {
  /**
   * Preprocesses an image by converting it to RGB color space, resizing it, and
   * normalizing pixel values.
   * @param bgr_image The input BGR image.
   * @param rgb_image The output RGB image.
   * @param img_shape The desired shape (width and height) of the image.
   * @return The preprocessed image as a normalized floating-point matrix.
   */
  // Convert image from BGR to RGB color space
  cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

  // Resize the image to (img_shape, img_shape)
  cv::resize(rgb_image, rgb_image, cv::Size(img_shape, img_shape));

  cv::Mat normalized_image;
  // Convert the image to floating point representation and normalize pixel
  // values
  rgb_image.convertTo(normalized_image, CV_32FC3, 1.0 / 255.0);

  return normalized_image;
}

void ImageHelper::DrawOnBBs(cv::Mat &bgr_image, int x1, int y1, int x2, int y2,
                            float score) {
  /**
   * Draws a bounding box and a label with the given score on the input image.
   * @param bgr_image The input BGR image.
   * @param x1 The x-coordinate of the top-left corner of the bounding box.
   * @param y1 The y-coordinate of the top-left corner of the bounding box.
   * @param x2 The x-coordinate of the bottom-right corner of the bounding box.
   * @param y2 The y-coordinate of the bottom-right corner of the bounding box.
   * @param score The score associated with the bounding box.
   */
  // Draw bounding box on the bgr_image
  cv::rectangle(bgr_image, cv::Point(x1, y1), cv::Point(x2, y2),
                cv::Scalar(0, 255, 0), 2);

  // Draw label on the bgr_image
  std::string label_text = "Score: " + std::to_string(score);
  int baseline;
  cv::Size label_size =
      cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
  cv::rectangle(
      bgr_image, cv::Point(x1, y1 - label_size.height - baseline - 10),
      cv::Point(x1 + label_size.width, y1), cv::Scalar(255, 255, 255), -1);
  cv::putText(bgr_image, label_text, cv::Point(x1, y1 - baseline - 5),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
}

void ImageHelper::DrawFps(cv::Mat &bgr_image, float fps) {
  /**
   * Draws the frames per second (FPS) on the input image.
   * @param bgr_image The input BGR image.
   * @param fps The frames per second value to be displayed.
   */
  // Draw FPS on the image
  std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
  int baseline;
  cv::Size fps_size =
      cv::getTextSize(fps_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
  cv::rectangle(bgr_image, cv::Point(10, 10),
                cv::Point(10 + fps_size.width, 10 + fps_size.height + baseline),
                cv::Scalar(255, 255, 255), -1);
  cv::putText(bgr_image, fps_text, cv::Point(10, 10 + fps_size.height),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
}
