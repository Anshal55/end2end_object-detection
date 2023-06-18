#ifndef IM_HELPER
#define IM_HELPER

// GLOG
#include "glog/logging.h"
#include "glog/stl_logging.h"

// opencv dependencies
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

class ImageHelper {
private:
  /* data */
public:
  ImageHelper();
  ~ImageHelper();

  cv::Mat PreProcessImages(cv::Mat &bgr_image, cv::Mat &rgb_image,
                           int img_shape = 416);

  void DrawOnBBs(cv::Mat &bgr_image, int x1, int y1, int x2, int y2,
                 float score);

  void DrawFps(cv::Mat &bgr_image, float fps);
};

#endif // !IM_HELPER