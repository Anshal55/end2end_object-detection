#ifndef NMS
#define NMS

#include "src/includes/object_detection/bounding_box.h"
#include <algorithm>
#include <vector>

/**
 * Calculates the Intersection over Union (IoU) between two bounding boxes.
 * @param a The first bounding box.
 * @param b The second bounding box.
 * @return The IoU value between the two bounding boxes.
 */
float IoU(const BoundingBox &a, const BoundingBox &b) {
  float xmin = std::max(a.xmin, b.xmin);
  float ymin = std::max(a.ymin, b.ymin);
  float xmax = std::min(a.xmax, b.xmax);
  float ymax = std::min(a.ymax, b.ymax);

  float intersection =
      std::max(0.0f, xmax - xmin) * std::max(0.0f, ymax - ymin);
  float area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin);
  float area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin);

  return intersection / (area_a + area_b - intersection);
}

/**
 * Performs Non-Maximum Suppression (NMS) on a vector of bounding boxes.
 * @param boxes The vector of bounding boxes to perform NMS on.
 * @param iou_threshold The IoU threshold for overlapping bounding boxes.
 * @return The selected bounding boxes after NMS.
 */
std::vector<BoundingBox>
NonMaximumSuppression(const std::vector<BoundingBox> &boxes,
                      float iou_threshold) {
  // Sort the bounding boxes by their confidence scores in descending order
  std::vector<BoundingBox> sorted_boxes(boxes);
  std::sort(sorted_boxes.begin(), sorted_boxes.end(),
            [](const BoundingBox &a, const BoundingBox &b) {
              return a.confidence > b.confidence;
            });

  std::vector<BoundingBox> selected_boxes;
  while (!sorted_boxes.empty()) {
    // Select the bounding box with the highest confidence score
    selected_boxes.push_back(sorted_boxes.back());
    sorted_boxes.pop_back();

    // Remove the remaining bounding boxes that overlap significantly with
    // the selected bounding box
    for (auto it = sorted_boxes.begin(); it != sorted_boxes.end();) {
      if (IoU(selected_boxes.back(), *it) > iou_threshold) {
        it = sorted_boxes.erase(it);
      } else {
        ++it;
      }
    }
  }

  return selected_boxes;
}

#endif // !NMS