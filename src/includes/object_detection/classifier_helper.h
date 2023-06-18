#ifndef CL_HELPER
#define CL_HELPER

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

class ClassifierHelper {
private:
  std::unique_ptr<tflite::FlatBufferModel> classifier_model;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  TfLiteTensor *input_tensor_;

public:
  ClassifierHelper();
  ~ClassifierHelper();

public:
  const bool RunInference();

  void UpdateTensor(cv::Mat &input_image, int xmin, int ymin, int xmax,
                       int ymax);
};

#endif // !CL_HELPER