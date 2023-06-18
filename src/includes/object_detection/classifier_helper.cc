#include "src/includes/object_detection/classifier_helper.h"

ClassifierHelper::ClassifierHelper() {

  classifier_model = tflite::FlatBufferModel::BuildFromFile(
      "src/includes/object_detection/models/classifier.tflite");

  if (!classifier_model) {
    LOG(FATAL) << "Failed to load classifier model";
  }

  // Build the interpreter_
  tflite::InterpreterBuilder builder(*classifier_model, resolver);
  builder(&interpreter_);
  if (!interpreter_) {
    LOG(FATAL) << "Failed to construct interpreter_";
  }

  // Allocate tensor buffers.
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors";
  }

  // Get input tensor
  input_tensor_ = interpreter_->input_tensor(0);

  LOG(INFO)
      << "****************** Classifier laoded and and Interpreter built. "
         "******************";
}

ClassifierHelper::~ClassifierHelper() {
  LOG(INFO) << "Closing ClassifierHelper";
}

void ClassifierHelper::UpdateTensor(cv::Mat &input_image, int xmin, int ymin,
                                    int xmax, int ymax) {
  // Create a copy of the original image using clone()
  cv::Mat temp_image = input_image.clone();
  // Convert image from BGR to RGB
  cv::cvtColor(temp_image, temp_image, cv::COLOR_BGR2RGB);

  // Ensure xmin, ymin, xmax, ymax are within the image bounds
  xmin = std::max(0, std::min(xmin, temp_image.cols - 1));
  ymin = std::max(0, std::min(ymin, temp_image.rows - 1));
  xmax = std::max(0, std::min(xmax, temp_image.cols - 1));
  ymax = std::max(0, std::min(ymax, temp_image.rows - 1));

  std::cout << xmin << ", " << ymin << ", " << xmax << ", " << ymax;

  // define roi
  cv::Rect roi(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1));

  std::cout << "Before crop\n";

  // Crop the image using the roi
  cv::Mat cropped_image = temp_image(roi);

  std::cout << "After Crop\n";

  // resize the image
  // Resize image
  cv::resize(cropped_image, cropped_image, cv::Size(192, 192));

  // Update the input_tensor_
  float *input = interpreter_->typed_input_tensor<float>(0);
  for (int i = 0; i < cropped_image.total() * cropped_image.channels(); ++i) {
    input[i] = cropped_image.data[i] / 255.0;
  }
}

const bool ClassifierHelper::RunInference() {
  LOG(INFO) << "Inference Runner Called.";

  auto start_time_infer = std::chrono::high_resolution_clock::now();

  // Run inference
  if (interpreter_->Invoke() != kTfLiteOk) {
    LOG(FATAL) << "Failed to invoke tflite!";
  }

  auto end_time_infer = std::chrono::high_resolution_clock::now();
  double elapsed_time_infer =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time_infer -
                                                            start_time_infer)
          .count();

  LOG(INFO) << "Classification taken per inference = " << elapsed_time_infer
            << " ms";

  // Get output tensor
  TfLiteTensor *output_tensor = interpreter_->output_tensor(0);

  // Read output buffer
  float sigmoid_score = output_tensor->data.f[0];

  LOG(INFO) << "Sigmoid score: " << sigmoid_score;

  return sigmoid_score > 0.85 ? true : false;
}