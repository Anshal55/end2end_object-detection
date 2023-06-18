#ifndef OD_HELPER
#define OD_HELPER

#include <iostream>

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
#include "tensorflow/lite/optional_debug_tools.h"

#include "src/includes/object_detection/bounding_box.h"

class ObjectDetectionHelper {
private:
  std::unique_ptr<tflite::FlatBufferModel> od_model;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  TfLiteTensor *input_tensor_;

  int input_index;
  TfLiteIntArray *input_dims;
  int output_index;
  TfLiteIntArray *output_dims;

public:
  ObjectDetectionHelper();
  ~ObjectDetectionHelper();

public:
  const bool RunInference();

  void UpdateTensor(cv::Mat &input_image);

  std::vector<BoundingBox>
  GatherBoundingBoxes(float confidence_threshold = 0.4);
};

#endif // !OD_HELPER