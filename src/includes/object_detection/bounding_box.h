#ifndef B_BOX
#define B_BOX

// Structure to hold the Boundind Boxes
struct BoundingBox {
  float xmin, ymin, xmax, ymax;
  float confidence;
};

#endif // !B_BOX