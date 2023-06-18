#include "src/includes/object_detection/object_detection_helper.h"

ObjectDetectionHelper::ObjectDetectionHelper() {

  od_model = tflite::FlatBufferModel::BuildFromFile(
      "src/includes/object_detection/models/main_model_ssd_mobilenet.tflite");

  if (!od_model) {
    LOG(FATAL) << "Failed to load classifier model";
  }

  // Build the interpreter_
  tflite::InterpreterBuilder builder(*od_model, resolver);
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

  // Get input and output tensors
  input_index = interpreter_->inputs()[0];
  input_dims = interpreter_->tensor(input_index)->dims;
  output_index = interpreter_->outputs()[0];
  output_dims = interpreter_->tensor(output_index)->dims;

  LOG(INFO)
      << "****************** Object Detector laoded and and Interpreter built. "
         "******************";
}

ObjectDetectionHelper::~ObjectDetectionHelper() {
  LOG(INFO) << "Closing ObjectDetectionHelper";
}

void ObjectDetectionHelper::UpdateTensor(cv::Mat &input_image) {
  // Fill input buffer with data
  float *input_data = interpreter_->typed_tensor<float>(input_index);

  // copy data to tensor
  memcpy(input_data, input_image.data,
         sizeof(float) * input_image.total() * input_image.channels());
}

const bool ObjectDetectionHelper::RunInference() {
  LOG(INFO) << "Inference Runner Called.";

  auto start_time_infer = std::chrono::high_resolution_clock::now();

  // Run inference
  if (interpreter_->Invoke() != kTfLiteOk) {
    LOG(FATAL) << "Failed to invoke tflite!";
    return false;
  }

  auto end_time_infer = std::chrono::high_resolution_clock::now();
  double elapsed_time_infer =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time_infer -
                                                            start_time_infer)
          .count();

  LOG(INFO) << "Object Detection taken per inference = " << elapsed_time_infer
            << " ms";

  return true;
}

std::vector<BoundingBox>
ObjectDetectionHelper::GatherBoundingBoxes(float confidence_threshold) {
  //! Deal with the outputs and return a bounding box
  // Get number of outputs
  int output_count = interpreter_->outputs().size();

  // Get the number of detections
  float *output_data = interpreter_->typed_output_tensor<float>(2);
  int count = static_cast<int>(*output_data);

  // intermediate outputs
  std::vector<float> confidence_scores;
  std::vector<std::vector<float>> bbs;

  // Loop over all outputs
  for (int i = 0; i < (output_count - 2); ++i) {
    // Get output tensor
    TfLiteTensor *output_tensor = interpreter_->output_tensor(i);

    // Get pointer to output data
    float *output_data = interpreter_->typed_output_tensor<float>(i);

    // Get number of elements in output tensor
    int num_elements = output_tensor->bytes / sizeof(float);

    // Fill vectors with appropriate values
    if (i == 0) {
      confidence_scores.assign(output_data, output_data + num_elements);
    } else if (i == 1) {
      for (int j = 0; j < num_elements / 4; ++j) {
        std::vector<float> box(output_data + j * 4, output_data + (j + 1) * 4);
        bbs.push_back(box);
      }
    }
  }

  // Final output declaration
  std::vector<BoundingBox> bounding_boxes;

  // fill the final output based on the confidence threshold score
  for (int i = 0; i < count; ++i) {

    if (confidence_scores.at(i) < confidence_threshold)
      continue;

    BoundingBox bb;
    bb.confidence = confidence_scores.at(i);
    bb.ymin = bbs.at(i).at(0);
    bb.xmin = bbs.at(i).at(1);
    bb.ymax = bbs.at(i).at(2);
    bb.xmax = bbs.at(i).at(3);

    bounding_boxes.push_back(bb);
  }

  return bounding_boxes;
}